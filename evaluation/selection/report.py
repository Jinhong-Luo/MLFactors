"""因子评估汇总报告。"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from data.schema import Col
from evaluation.selection.ic import (
    calc_ic_series,
    calc_icir,
    calc_t_stat,
    calc_turnover,
    calc_ic_decay,
)
from evaluation.selection.layered import LayeredResult, layered_backtest
import numpy as np
from scipy import stats

REBALANCE_ICIR_PERIODS = {
    "week": 5,
    "half_month": 10,
    "month": 21,
}


def normalize_signal_dates(
        signal_dates: pd.Index | pd.Series | pd.DataFrame | None,
    ) -> pd.DatetimeIndex | None:
    if signal_dates is None:
        return None
    if isinstance(signal_dates, pd.DataFrame):
        values = signal_dates[Col.DATE] if Col.DATE in signal_dates.columns else signal_dates.index
    elif isinstance(signal_dates, pd.Series):
        values = signal_dates
    else:
        values = signal_dates
    return pd.DatetimeIndex(pd.to_datetime(values)).tz_localize(None).unique().sort_values()

def calc_factors_returns(
    market_data: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    buy_price_col: str = Col.OPEN,
    sell_price_col: str = Col.CLOSE,
) -> tuple[pd.Series, pd.Series]:
    buy_price = market_data[buy_price_col].unstack(Col.SYMBOL).sort_index()
    sell_price = market_data[sell_price_col].unstack(Col.SYMBOL).sort_index()
    buy_price.index = pd.DatetimeIndex(pd.to_datetime(buy_price.index)).tz_localize(None)
    sell_price.index = pd.DatetimeIndex(pd.to_datetime(sell_price.index)).tz_localize(None)
    trading_dates = buy_price.index

    buy_positions = trading_dates.searchsorted(signal_dates, side="right")
    buy_dates = pd.Series(pd.NaT, index=signal_dates, dtype="datetime64[ns]", name="rebalance_date")
    valid_buy = buy_positions < len(trading_dates)
    buy_dates.iloc[valid_buy] = trading_dates.take(buy_positions[valid_buy]).to_numpy()

    sell_dates = pd.Series(signal_dates, index=signal_dates, dtype="datetime64[ns]").shift(-1)
    returns = pd.DataFrame(index=signal_dates, columns=buy_price.columns, dtype=float)
    returns.index.name = Col.DATE
    returns.columns.name = Col.SYMBOL

    valid = buy_dates.notna() & sell_dates.notna() & (buy_dates <= sell_dates)
    if valid.any():
        valid_signal_dates = signal_dates[valid.to_numpy()]
        entry_price = buy_price.reindex(pd.DatetimeIndex(buy_dates.loc[valid_signal_dates]))
        exit_price = sell_price.reindex(pd.DatetimeIndex(sell_dates.loc[valid_signal_dates]))
        entry_price.index = valid_signal_dates
        exit_price.index = valid_signal_dates
        returns.loc[valid_signal_dates] = exit_price.divide(entry_price) - 1.0

    return returns.stack().rename("fwd_ret")


class FactorReport:
    """聚合全部评估指标，生成因子测试报告。

    Parameters
    ----------
    factor_values : 因子值 pd.Series，MultiIndex(date, symbol)，允许前置 NaN
    market_data : 行情数据 pd.DataFrame，MultiIndex(date, symbol)
    signal_dates : 因子生成日期序列；传入时按真实调仓日计算收益
    ic_method : "rank" (Spearman) 或 "pearson"
    n_groups : 分层回测分组数
    rebalance : 调仓周期，用于设置 ICIR 年化周期参数
    open_col : 开盘价列名，用于计算前向收益（T+1 开盘买入）
    """

    def __init__(
        self,
        factor_values: pd.Series,
        market_data: pd.DataFrame,
        signal_dates: pd.Index | pd.Series | pd.DataFrame | None = None,
        ic_method: str = "rank",
        n_groups: int = 5,
        rebalance: str | None = None,
        open_col: str = "open"
    ) -> None:
        self.factor_values = factor_values.dropna()
        self.market_data = market_data
        self.signal_dates = normalize_signal_dates(signal_dates)
        self.ic_method = ic_method
        self.n_groups = n_groups
        self.rebalance = rebalance
        self.forward_period = self._resolve_forward_period(rebalance)
        self.open_col = open_col

        if self.signal_dates.empty:
            raise ValueError("signal_dates 不能为空，至少需要一个调仓日期以计算前向收益")

        self.fwd_returns = calc_factors_returns(
            market_data,
            self.signal_dates,
            self.open_col,
        )

        # 缓存结果
        self._ic_series: pd.Series | None = None
        self._layered: LayeredResult | None = None
        self._summary: pd.DataFrame | None = None
        self._rebalance_holdings: pd.DataFrame | None = None

    @staticmethod
    def _resolve_forward_period(rebalance: str | None) -> int:
        if rebalance is None:
            return 1
        if rebalance not in REBALANCE_ICIR_PERIODS:
            raise ValueError(f"不支持的调仓周期: {rebalance}")
        return REBALANCE_ICIR_PERIODS[rebalance]

    # ------------------------------------------------------------------ #
    #  IC 相关
    # ------------------------------------------------------------------ #

    def ic_series(self) -> pd.Series:
        if self._ic_series is None:
            self._ic_series = calc_ic_series(
                self.factor_values, self.fwd_returns, self.ic_method
            )
        return self._ic_series

    def icir(self) -> float:
        return calc_icir(
            self.ic_series(),
            period=self.forward_period,
            annualize=True
        )

    def t_stat(self) -> tuple[float, float]:
        return calc_t_stat(self.ic_series())

    def turnover(self) -> pd.Series:
        return calc_turnover(self.factor_values)

    def rebalance_holdings(self) -> pd.DataFrame:
        """返回每次调仓的多头分组股票列表。"""
        if self._rebalance_holdings is not None:
            return self._rebalance_holdings

        combined = pd.DataFrame({
            "factor": self.factor_values,
            "returns": self.fwd_returns,
        }).dropna()
        if combined.empty:
            self._rebalance_holdings = pd.DataFrame(
                columns=["date", "group", "symbol_count", "symbols"]
            )
            return self._rebalance_holdings

        records = []
        dates = combined.index.get_level_values(0).unique().sort_values()
        target_group = self.n_groups
        for dt in dates:
            cross = combined.xs(dt, level=0).copy()
            if len(cross) < self.n_groups:
                continue

            ranks = cross["factor"].rank(method="first")
            groups = pd.qcut(ranks, self.n_groups, labels=False, duplicates="drop") + 1
            selected = cross.loc[groups == target_group].sort_values("factor", ascending=False)
            if selected.empty:
                continue

            symbols = selected.index.get_level_values(-1).astype(str).tolist()
            records.append({
                "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                "group": int(target_group),
                "symbol_count": int(len(symbols)),
                "symbols": ", ".join(symbols),
            })

        self._rebalance_holdings = pd.DataFrame.from_records(
            records,
            columns=["date", "group", "symbol_count", "symbols"],
        )
        return self._rebalance_holdings

    def ic_decay(self, max_lag: int = 20) -> pd.Series:
        return calc_ic_decay(
            self.factor_values,
            self.fwd_returns,
            max_lag=max_lag,
            method=self.ic_method,
        )

    # ------------------------------------------------------------------ #
    #  分层回测
    # ------------------------------------------------------------------ #

    def layered(self) -> LayeredResult:
        if self._layered is None:
            self._layered = layered_backtest(
                self.factor_values, self.fwd_returns, self.n_groups, period=self.forward_period
            )
        return self._layered

    # ------------------------------------------------------------------ #
    #  汇总
    # ------------------------------------------------------------------ #

    def summary(self) -> pd.DataFrame:
        """生成各周期的汇总指标表。"""
        if self._summary is not None:
            return self._summary

        records = []
        turnover_mean = float(self.turnover().mean())

        ic_s = self.ic_series()
        ic_mean = ic_s.mean()
        ic_std = ic_s.std()
        
        # 使用折现惩罚修复 ICIR 和 t_stat 虚高
        icir = calc_icir(ic_s, self.forward_period, annualize=True)

        t, pval = calc_t_stat(ic_s)
        df = len(ic_s.dropna()) - 1
        if df > 0:
            pval = stats.t.sf(np.abs(t), df) * 2
        else:
            pval = np.nan 
        ic_positive_ratio = (ic_s > 0).mean()

        lr = self.layered()
        long_sharpe = 0.0
        if not lr.sharpe_ratios.empty:
            top_group = lr.sharpe_ratios.index.max()
            long_sharpe = float(lr.sharpe_ratios.loc[top_group])

        records.append({
            "period": 1,
            "IC_mean": round(ic_mean, 4),
            "IC_std": round(ic_std, 4),
            "ICIR": round(icir, 4),
            "t_stat": round(t, 4),
            "p_value": round(pval, 6),
            "IC>0_ratio": round(ic_positive_ratio, 4),
            "turnover": round(turnover_mean, 4),
            "long_sharpe": round(long_sharpe, 4),
            "long_max_drawdown": round(lr.long_max_drawdown, 4),
            "short_max_drawdown": round(lr.short_max_drawdown, 4),
            "top_excess_annual": round(lr.top_excess_annual, 4),
            "top_excess_max_dd": round(lr.top_excess_max_drawdown, 4),
            "top_excess_calmar": round(lr.top_excess_calmar, 4),
        })

        self._summary = pd.DataFrame(records).set_index("period")
        return self._summary

    def to_dict(self) -> dict:
        return self.summary().to_dict(orient="index")

    def print(self) -> None:
        logger.info("=" * 60)
        logger.info("因子评估报告: {}", getattr(self.factor_values, "name", "unknown"))
        logger.info("=" * 60)
        summary = self.summary()
        print(summary.to_string())
        logger.info("-" * 60)
