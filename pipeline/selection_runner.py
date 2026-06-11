"""端到端因子挖掘流水线。"""

from __future__ import annotations

from typing import Type

import pandas as pd
from loguru import logger

from data.base import DataLoader
from data.schema import Col
from evaluation.selection.report import FactorReport
from evaluation.plot import plot_factor_report
from factors.base import BaseFactor
from factors.registry import FactorRegistry
from models.base import BaseModel


class SelectionPipeline:
    """端到端因子挖掘流水线。

    支持两种模式:
    1. 纯因子评估模式 — 直接计算因子并评估 IC/ICIR/分层回测
    2. ML模型模式 — 用多个因子作为特征训练模型，模型预测值作为综合 alpha

    用法::

        pipeline = FactorPipeline()
        pipeline.set_data_loader(LocalLoader(market_path="data.csv"))
        pipeline.add_factors(["momentum_5", "volatility_20"])
        # 模式1: 纯因子评估
        results = pipeline.run()

        # 模式2: ML模型
        pipeline.set_model(TreeModel(engine="lgbm"))
        results = pipeline.run()
    """

    def __init__(
        self,
        ic_method: str = "rank",
        n_groups: int = 5,
        rebalance: str | None = None,
    ) -> None:
        self._loader: DataLoader | None = None
        self._factor_names: list[str] = []
        self._factor_instances: list[BaseFactor] = []
        self._model: BaseModel | None = None

        self._data: dict[str, pd.DataFrame] | None = None

        self._ic_method = ic_method
        self._n_groups = n_groups
        self._rebalance = rebalance

        self.aggregated_market_data: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    #  配置方法（链式调用）
    # ------------------------------------------------------------------ #

    def set_data_loader(self, loader: DataLoader) -> SelectionPipeline:
        self._loader = loader
        return self

    def add_factors(
        self,
        factors: list[str | Type[BaseFactor] | BaseFactor],
    ) -> SelectionPipeline:
        """添加因子。

        Parameters
        ----------
        factors : 因子名称列表、因子类列表、或因子实例列表
        """
        for f in factors:
            if isinstance(f, str):
                self._factor_names.append(f)
            elif isinstance(f, type) and issubclass(f, BaseFactor):
                self._factor_instances.append(f())
            elif isinstance(f, BaseFactor):
                self._factor_instances.append(f)
            else:
                raise TypeError(f"不支持的因子类型: {type(f)}")
        return self

    def set_model(self, model: BaseModel) -> SelectionPipeline:
        self._model = model
        return self

    def load_data(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> SelectionPipeline:
        """手动加载数据（也可由 run() 自动调用）。"""
        if self._loader is None:
            raise RuntimeError("请先调用 set_data_loader()")

        logger.info("加载数据...")
        self._data = self._loader.load_data(symbols, start, end)
        if self._rebalance is not None:
            market_data = self._data.get("market")
            if market_data is not None:
                self.aggregated_market_data = self.aggregate_market_data(market_data, self._rebalance)
                self._data["signal_dates"] = pd.DatetimeIndex(
                    self.aggregated_market_data.index.get_level_values(Col.DATE).unique()
                ).sort_values()
        # if self._requires_alpha158_data() and hasattr(self._loader, "load_alpha158_data"):
        #     self._data["alpha158"] = self._loader.load_alpha158_data(symbols, start, end)
        if start is not None and hasattr(self._loader, "load_predata"):
            logger.info("加载预热数据...")
            predata = self._loader.load_predata(symbols, start, end)
            self._data = self._prepend_predata(self._data, predata)

        for table_name, table_data in self._data.items():
            logger.info("{} 数据: {} 行", table_name, len(table_data))
        return self

    # ------------------------------------------------------------------ #
    #  执行
    # ------------------------------------------------------------------ #

    def _requires_alpha158_data(self) -> bool:
        names = set(self._factor_names)
        names.update(factor.name for factor in self._factor_instances)
        return "svm_alpha158" in names

    @staticmethod
    def _prepend_predata(
        data: dict[str, pd.DataFrame],
        predata: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        merged_data = dict(data)
        for table_name, table_data in predata.items():
            if table_name == "signal_dates" or not isinstance(table_data, pd.DataFrame) or table_data.empty:
                continue

            current_data = merged_data.get(table_name)
            if isinstance(current_data, pd.DataFrame) and not current_data.empty:
                combined = pd.concat([table_data, current_data]).sort_index()
                if combined.index.has_duplicates:
                    combined = combined[~combined.index.duplicated(keep="last")]
                merged_data[table_name] = combined
            else:
                merged_data[table_name] = table_data
        return merged_data

    def run(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        show_plot: bool = True,
    ) -> dict[str, FactorReport] | dict:
        """执行完整流水线。

        Returns
        -------
        - 纯因子模式: dict[factor_name, FactorReport]
        - ML模型模式: dict with keys "model_report", "cv_result", "feature_importance"
        """
        # 1. 加载数据
        if self._data is None:
            self.load_data(symbols, start, end)

        if self._data is None:
            raise RuntimeError("数据未加载")
        if self._data.get("market") is None or self._data["market"].empty:
            raise RuntimeError("market 数据为空")

        # 2. 收集因子实例
        all_factors: list[BaseFactor] = list(self._factor_instances)
        for name in self._factor_names:
            cls = FactorRegistry.get(name)
            all_factors.append(cls())

        if not all_factors:
            raise RuntimeError("未添加任何因子，请先调用 add_factors()")

        # 3. 计算因子值
        logger.info("计算 {} 个因子...", len(all_factors))
        factor_values: dict[str, pd.Series] = {}
        for f in all_factors:
            logger.info("  计算 {} ...", f.name)
            factor_values[f.name] = f.generate_signals(self._data).stack(future_stack=True).rename(f.name)

        factor_df = pd.DataFrame(factor_values)

        # 4. 执行评估
        return self._run_factor_evaluation(factor_df, show_plot)

    def _run_factor_evaluation(
        self,
        factor_df: pd.DataFrame,
        show_plot: bool,
    ) -> dict[str, FactorReport]:
        """纯因子评估模式。"""
        if self._data is None:
            raise RuntimeError("数据未加载")

        ic_method = self._ic_method
        n_groups = self._n_groups

        reports: dict[str, FactorReport] = {}

        for col in factor_df.columns:
            logger.info("评估因子: {}", col)
            factor_s = factor_df[col].dropna()

            report = FactorReport(
                factor_values=factor_s,
                market_data=self._data.get("market"),
                signal_dates=self._data.get("signal_dates") if self._rebalance is not None else None,
                ic_method=ic_method,
                n_groups=n_groups,
                rebalance=self._rebalance,
            )
            report.print()
            reports[col] = report

            if show_plot:
                try:
                    ic_s = report.ic_series()
                    lr = report.layered()
                    decay = report.ic_decay()
                    plot_factor_report(ic_s, lr, decay, factor_name=col)
                except Exception as e:
                    logger.warning("绘图失败: {}", e)

        return reports

    @staticmethod
    def _rebalance_key(dates: pd.Series, rebalance: str) -> pd.Series:
        dates = pd.to_datetime(dates).dt.tz_localize(None)
        if rebalance == "week":
            return dates.dt.to_period("W-SUN").astype(str)
        if rebalance == "month":
            return dates.dt.to_period("M").astype(str)
        if rebalance == "half_month":
            half = dates.dt.day.gt(15).astype(int) + 1
            return dates.dt.to_period("M").astype(str) + "-H" + half.astype(str)
        raise ValueError(f"不支持的调仓周期: {rebalance}")

    @classmethod
    def aggregate_market_data(cls, market_data: pd.DataFrame, rebalance: str) -> pd.DataFrame:
        """按自然周、自然半月或自然月聚合 OHLCV 行情。"""
        if market_data.empty:
            return market_data.copy()

        df = market_data.reset_index().copy()
        df[Col.DATE] = pd.to_datetime(df[Col.DATE]).dt.tz_localize(None)
        df = df.sort_values([Col.SYMBOL, Col.DATE])
        df["_period"] = cls._rebalance_key(df[Col.DATE], rebalance)
        period_end_dates = df.groupby("_period", observed=True)[Col.DATE].max()

        agg_map: dict[str, str] = {}
        for column, method in (
            (Col.DATE, "last"),
            (Col.OPEN, "first"),
            (Col.HIGH, "max"),
            (Col.LOW, "min"),
            (Col.CLOSE, "last"),
            (Col.VOLUME, "sum"),
            (Col.ADJ_CLOSE, "last"),
        ):
            if column in df.columns:
                agg_map[column] = method

        aggregated = (
            df.groupby([Col.SYMBOL, "_period"], observed=True)
            .agg(agg_map)
            .reset_index()
        )
        aggregated["_period_end"] = aggregated["_period"].map(period_end_dates)
        abnormal = aggregated[Col.DATE].ne(aggregated["_period_end"])
        if abnormal.any():
            logger.warning(
                "剔除非周期最后交易日的聚合数据: {} 行（rebalance={}）",
                int(abnormal.sum()),
                rebalance,
            )
            aggregated = aggregated.loc[~abnormal].copy()
        aggregated = aggregated.drop(columns=["_period", "_period_end"])
        required = [column for column in (Col.OPEN, Col.HIGH, Col.LOW, Col.CLOSE) if column in aggregated.columns]
        if required:
            aggregated = aggregated.dropna(subset=required, how="any")
        aggregated = aggregated.set_index([Col.DATE, Col.SYMBOL]).sort_index()
        aggregated.index.names = [Col.DATE, Col.SYMBOL]
        return aggregated
