"""基于市场变量的有效因子序数回归轮动。"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from data.schema import Col, FundamentalCol
from factors.base import BaseFactor
from factors.registry import register_factor
from . import alpha158_kline, alpha158_price, alpha158_rolling

try:
    from statsmodels.miscmodels.ordinal_model import OrderedModel
except ModuleNotFoundError:
    OrderedModel = None


def _alpha158_factor_class_map() -> dict[str, type[BaseFactor]]:
    return {
        "alpha158_std60": getattr(alpha158_rolling, "Alpha158Std60"),
        "alpha158_klen": alpha158_kline.Alpha158Klen,
        "alpha158_std20": getattr(alpha158_rolling, "Alpha158Std20"),
        "alpha158_std30": getattr(alpha158_rolling, "Alpha158Std30"),
        "alpha158_kup": alpha158_kline.Alpha158Kup,
        "alpha158_high0": alpha158_price.Alpha158High0,
        "alpha158_std10": getattr(alpha158_rolling, "Alpha158Std10"),
        "alpha158_std5": getattr(alpha158_rolling, "Alpha158Std5"),
        "alpha158_max5": getattr(alpha158_rolling, "Alpha158Max5"),
        "alpha158_max10": getattr(alpha158_rolling, "Alpha158Max10"),
    }


@register_factor
class OrdinalFactorRotation(BaseFactor):
    name = "ordinal_factor_rotation"
    description = "基于市场变量的有效因子序数回归轮动"
    category = "composite"

    ETF_SYMBOLS = [
        "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU",
        "XLRE", "XLC", "SPY", "QQQ", "IWM", "DIA",
    ]
    GROWTH_ETFS = ["XLK", "XLY", "XLC"]
    DEFENSIVE_ETFS = ["XLP", "XLU", "XLV"]
    CYCLICAL_ETFS = ["XLF", "XLI", "XLE", "XLB"]

    FACTOR_NAMES = [
        "roe",
        "delta_roa",
        "delta_roe",
        "delta_eps",
        "revenue_growth",
        "pb",
        "reversal_1m",
        "reversal_3m",
        "reversal_6m",
        "dif",
        "dea",
    ]
    TRAIN_WINDOW = 252
    TOP_QUANTILE = 0.2
    TOP_FRAC = 0.5
    # 当前版本保持日频信号，仅每 WEIGHT_UPDATE_INTERVAL 个交易日更新一次权重。
    WEIGHT_UPDATE_INTERVAL = 21
    MONTHLY_TRAIN_WINDOW = 60
    MONTHLY_WEIGHT_UPDATE_INTERVAL = 1
    FUNDAMENTAL_DELTA_WINDOW = 63
    MIN_AVAILABLE_FACTORS = 5
    MIN_STOCK_FACTORS = 3

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        close_all_daily = market_data[Col.CLOSE].unstack(Col.SYMBOL).sort_index()

        monthly_market = self._to_monthly_market_data(market_data)
        close_all = monthly_market[Col.CLOSE].unstack(Col.SYMBOL).sort_index()
        volume_all = self._unstack_optional(monthly_market, Col.VOLUME)
        if volume_all is not None:
            volume_all = volume_all.reindex(index=close_all.index, columns=close_all.columns)

        # DataFrame 列索引必须保留原始 symbol 对象，ETF 判断只在比较时转 str。
        all_symbols = list(close_all.columns)
        etf_symbols = [symbol for symbol in all_symbols if self._is_etf_symbol(str(symbol))]
        stock_symbols = [symbol for symbol in all_symbols if symbol not in etf_symbols]
        if not stock_symbols:
            raise ValueError("ordinal_factor_rotation 无可用股票标的")

        close_daily = close_all_daily[stock_symbols]
        close = close_all[stock_symbols]
        daily_factor_dict = self._compute_sub_factors(market_data, fundamental_data, close_daily)
        factor_dict = {
            name: factor.reindex(index=close.index, columns=close.columns)
            for name, factor in daily_factor_dict.items()
        }
        available = [name for name, factor in factor_dict.items() if not factor.dropna(how="all").empty]
        if len(available) < self.MIN_AVAILABLE_FACTORS:
            raise ValueError(
                f"ordinal_factor_rotation 可用子因子不足: {len(available)} < {self.MIN_AVAILABLE_FACTORS}"
            )

        # 月末因子值对应下一月月末 close-to-close 收益，用于月频序数回归。
        stock_returns_forward = close.pct_change(fill_method=None).shift(-1)
        market_features = self._build_monthly_market_features(close_all, volume_all, etf_symbols)
        factor_returns = self._compute_factor_returns(
            factor_dict,
            stock_returns_forward,
            top_quantile=self.TOP_QUANTILE,
        )
        factor_ranks = self._factor_returns_to_ranks(factor_returns)
        factor_weights = self._compute_dynamic_factor_weights(
            factor_returns,
            factor_ranks,
            market_features,
            train_window=self.MONTHLY_TRAIN_WINDOW,
            update_interval=self.MONTHLY_WEIGHT_UPDATE_INTERVAL,
        )
        signals = self._combine_factor_signals(factor_dict, factor_weights)
        signals = signals.reindex(index=close.index, columns=stock_symbols)
        signals.index.name = Col.DATE
        signals.columns.name = Col.SYMBOL

        self.last_monthly_market_data = monthly_market
        self.last_market_features = market_features
        self.last_factor_dict = factor_dict
        self.last_factor_returns = factor_returns
        self.last_factor_ranks = factor_ranks
        self.last_factor_weights = factor_weights
        return signals

    @classmethod
    def _is_etf_symbol(cls, symbol: str) -> bool:
        return str(symbol).upper() in {item.upper() for item in cls.ETF_SYMBOLS}

    def _compute_sub_factors(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None,
        close: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        blank = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        fundamentals = self._align_fundamentals(fundamental_data, close)

        roe = self._fundamental_panel(fundamentals, FundamentalCol.ROE, close)
        roa = self._fundamental_panel(fundamentals, FundamentalCol.ROA, close)
        eps = self._fundamental_panel(fundamentals, FundamentalCol.EPS, close)
        pb = self._fundamental_panel(fundamentals, FundamentalCol.PB, close)
        revenue_growth = self._fundamental_panel(fundamentals, FundamentalCol.REVENUE_GROWTH, close)
        revenue = self._first_available_fundamental(fundamentals, ["revenue", "operating_revenue"], close)

        raw_momentum_1m = close.pct_change(21, fill_method=None)
        raw_momentum_3m = close.pct_change(63, fill_method=None)
        raw_momentum_6m = close.pct_change(126, fill_method=None)
        dif = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(
            span=26,
            adjust=False,
            min_periods=26,
        ).mean()
        dea = dif.ewm(span=9, adjust=False, min_periods=9).mean()
        pb_clean = pb.where(pb > 0) if pb is not None else None

        factors = {
            "roe": roe if roe is not None else blank.copy(),
            # 基本面已前向填充到日频，1 日 diff 会大量为 0；63 日近似季度变化。
            "delta_roa": roa - roa.shift(self.FUNDAMENTAL_DELTA_WINDOW) if roa is not None else blank.copy(),
            "delta_roe": roe - roe.shift(self.FUNDAMENTAL_DELTA_WINDOW) if roe is not None else blank.copy(),
            "delta_eps": eps - eps.shift(self.FUNDAMENTAL_DELTA_WINDOW) if eps is not None else blank.copy(),
            "revenue_growth": revenue_growth if revenue_growth is not None else self._revenue_growth(revenue, blank),
            # PB <= 0 不是有效估值信号，避免 -PB 形成异常正向极值。
            "pb": -pb_clean if pb_clean is not None else blank.copy(),
            "reversal_1m": -raw_momentum_1m,
            "reversal_3m": -raw_momentum_3m,
            "reversal_6m": -raw_momentum_6m,
            "dif": dif,
            "dea": dea,
        }
        return {name: factor.reindex(index=close.index, columns=close.columns) for name, factor in factors.items()}

    def _build_market_features(
        self,
        close_all: pd.DataFrame,
        volume_all: pd.DataFrame | None,
        etf_symbols: list[object],
    ) -> pd.DataFrame:
        usable_etfs = [symbol for symbol in etf_symbols if symbol in close_all.columns]
        proxy = close_all[usable_etfs] if usable_etfs else close_all
        returns = proxy.pct_change(fill_method=None)
        market_return = returns.mean(axis=1)
        market_level = (1.0 + market_return.fillna(0.0)).cumprod()

        features = pd.DataFrame(index=close_all.index)
        for window in (5, 21, 63):
            features[f"market_ret_{window}d"] = market_level.pct_change(window, fill_method=None)
        for window in (21, 63):
            # market_vol 使用年化口径，便于不同窗口之间比较。
            features[f"market_vol_{window}d"] = (
                market_return.rolling(window, min_periods=max(5, window // 3)).std() * np.sqrt(252)
            )

        rolling_peak = market_level.rolling(63, min_periods=21).max()
        features["market_drawdown_63d"] = market_level / rolling_peak - 1.0
        for window in (21, 63):
            sector_ret = proxy.pct_change(window, fill_method=None)
            features[f"sector_dispersion_{window}d"] = sector_ret.std(axis=1)
            valid_count = sector_ret.notna().sum(axis=1)
            positive_count = sector_ret.gt(0).sum(axis=1)
            # 全 NaN 时表示没有可用板块收益，不能误记为 0% 上涨宽度。
            features[f"sector_breadth_{window}d"] = positive_count / valid_count.replace(0, np.nan)
            features[f"sector_avg_corr_{window}d"] = self._rolling_avg_corr(returns, window)
            features[f"top3_minus_bottom3_sector_ret_{window}d"] = self._top_bottom_spread(sector_ret)

        for window in (21, 63):
            features[f"growth_defensive_ret_{window}d"] = self._group_relative_return(
                close_all,
                self.GROWTH_ETFS,
                self.DEFENSIVE_ETFS,
                window,
            )
            features[f"cyclical_defensive_ret_{window}d"] = self._group_relative_return(
                close_all,
                self.CYCLICAL_ETFS,
                self.DEFENSIVE_ETFS,
                window,
            )

        if volume_all is not None:
            volume_proxy = volume_all[[symbol for symbol in usable_etfs if symbol in volume_all.columns]]
            if volume_proxy.empty:
                volume_proxy = volume_all
            total_volume = volume_proxy.sum(axis=1, min_count=1)
            vol_mean = total_volume.rolling(21, min_periods=10).mean()
            vol_std = total_volume.rolling(21, min_periods=10).std()
            features["market_volume_zscore_21d"] = (total_volume - vol_mean) / vol_std.replace(0, np.nan)

        return features.replace([np.inf, -np.inf], np.nan)

    def _compute_factor_returns(
        self,
        factor_dict: dict[str, pd.DataFrame],
        stock_returns_forward: pd.DataFrame,
        top_quantile: float = 0.2,
    ) -> pd.DataFrame:
        result = pd.DataFrame(index=stock_returns_forward.index, columns=list(factor_dict), dtype=float)
        for name, factor in factor_dict.items():
            aligned_factor = factor.reindex_like(stock_returns_forward)
            values = []
            for date, factor_row in aligned_factor.iterrows():
                returns_row = stock_returns_forward.loc[date]
                data = pd.DataFrame({"factor": factor_row, "return": returns_row}).dropna()
                if len(data) < 5:
                    values.append(np.nan)
                    continue
                count = max(1, int(np.floor(len(data) * top_quantile)))
                ordered = data.sort_values("factor")
                short_ret = ordered.head(count)["return"].mean()
                long_ret = ordered.tail(count)["return"].mean()
                values.append(long_ret - short_ret)
            result[name] = values
        return result

    @staticmethod
    def _factor_returns_to_ranks(factor_returns: pd.DataFrame) -> pd.DataFrame:
        return factor_returns.rank(axis=1, ascending=False, method="min")

    def _compute_dynamic_factor_weights(
        self,
        factor_returns: pd.DataFrame,
        factor_ranks: pd.DataFrame,
        market_features: pd.DataFrame,
        train_window: int | None = None,
        update_interval: int | None = None,
    ) -> pd.DataFrame:
        factor_returns = factor_returns.reindex(columns=self.FACTOR_NAMES)
        factor_ranks = factor_ranks.reindex(columns=self.FACTOR_NAMES)
        market_features = market_features.reindex(factor_returns.index)
        weights = pd.DataFrame(0.0, index=factor_returns.index, columns=self.FACTOR_NAMES)
        previous: pd.Series | None = None
        self._ordered_model_update_attempts = 0
        self._ordered_model_update_successes = 0
        train_window = self.TRAIN_WINDOW if train_window is None else train_window
        update_interval = self.WEIGHT_UPDATE_INTERVAL if update_interval is None else update_interval

        for position, date in enumerate(factor_returns.index):
            start = max(0, position - train_window)
            hist_returns = factor_returns.iloc[start:position]
            hist_ranks = factor_ranks.iloc[start:position]
            hist_features = market_features.iloc[start:position]
            current_features = market_features.loc[date]

            should_update = previous is None or position % update_interval == 0
            if not should_update:
                current = previous
            elif len(hist_returns.dropna(how="all")) < max(20, self.MIN_AVAILABLE_FACTORS):
                current = self._fallback_weights(hist_returns, previous)
            elif OrderedModel is None:
                current = self._fallback_weights(hist_returns, previous)
            else:
                self._ordered_model_update_attempts += 1
                current = self._ordered_model_weights(hist_returns, hist_ranks, hist_features, current_features)
                if current is not None:
                    self._ordered_model_update_successes += 1
                if current is None:
                    current = self._fallback_weights(hist_returns, previous)

            weights.loc[date] = current
            previous = current

        return weights

    def _combine_factor_signals(
        self,
        factor_dict: dict[str, pd.DataFrame],
        factor_weights: pd.DataFrame,
    ) -> pd.DataFrame:
        zscores = {
            name: self._cross_sectional_zscore(factor.reindex(factor_weights.index))
            for name, factor in factor_dict.items()
        }
        columns = next(iter(factor_dict.values())).columns
        signals = pd.DataFrame(np.nan, index=factor_weights.index, columns=columns)

        for date in factor_weights.index:
            weight_row = factor_weights.loc[date].reindex(self.FACTOR_NAMES).fillna(0.0)
            numerator = pd.Series(0.0, index=columns)
            denominator = pd.Series(0.0, index=columns)
            counts = pd.Series(0, index=columns)

            for name, weight in weight_row.items():
                if weight <= 0 or name not in zscores:
                    continue
                values = zscores[name].loc[date].reindex(columns)
                mask = values.notna()
                numerator.loc[mask] += values.loc[mask] * weight
                denominator.loc[mask] += weight
                counts.loc[mask] += 1

            valid = (denominator > 0) & (counts >= self.MIN_STOCK_FACTORS)
            signals.loc[date, valid] = numerator.loc[valid] / denominator.loc[valid]

        signals.index.name = Col.DATE
        signals.columns.name = Col.SYMBOL
        return signals

    @staticmethod
    def _unstack_optional(data: pd.DataFrame, column: str) -> pd.DataFrame | None:
        if column not in data.columns:
            return None
        return data[column].unstack(Col.SYMBOL).sort_index()

    @staticmethod
    def _align_fundamentals(
        fundamental_data: pd.DataFrame | None,
        close: pd.DataFrame,
    ) -> pd.DataFrame | None:
        if fundamental_data is None or fundamental_data.empty:
            return None
        aligned = fundamental_data.copy()
        if isinstance(aligned.index, pd.MultiIndex):
            dates = pd.to_datetime(aligned.index.get_level_values(Col.DATE))
            aligned = aligned.copy()
            aligned.index = pd.MultiIndex.from_arrays(
                [dates, aligned.index.get_level_values(Col.SYMBOL)],
                names=[Col.DATE, Col.SYMBOL],
            )
        return aligned.sort_index()

    def _fundamental_panel(
        self,
        fundamental_data: pd.DataFrame | None,
        column: str,
        close: pd.DataFrame,
    ) -> pd.DataFrame | None:
        if fundamental_data is None or column not in fundamental_data.columns:
            return None
        panel = fundamental_data[column].unstack(Col.SYMBOL)
        panel = panel.apply(pd.to_numeric, errors="coerce")
        if isinstance(panel.index, pd.DatetimeIndex) and isinstance(close.index, pd.DatetimeIndex):
            target_tz = close.index.tz
            source_tz = panel.index.tz
            if target_tz is not None and source_tz is None:
                panel.index = panel.index.tz_localize(target_tz)
            elif target_tz is None and source_tz is not None:
                panel.index = panel.index.tz_convert(None)
            elif target_tz is not None and source_tz is not None and target_tz != source_tz:
                panel.index = panel.index.tz_convert(target_tz)
        panel = panel.sort_index()
        full_index = panel.index.union(close.index)
        panel = panel.reindex(full_index).sort_index().ffill()
        return panel.reindex(index=close.index, columns=close.columns)

    def _first_available_fundamental(
        self,
        fundamental_data: pd.DataFrame | None,
        columns: list[str],
        close: pd.DataFrame,
    ) -> pd.DataFrame | None:
        for column in columns:
            panel = self._fundamental_panel(fundamental_data, column, close)
            if panel is not None:
                return panel
        return None

    def _revenue_growth(self, revenue: pd.DataFrame | None, blank: pd.DataFrame) -> pd.DataFrame:
        if revenue is None:
            return blank.copy()
        if revenue.dropna(how="all").empty:
            return blank.copy()
        return revenue.pct_change(self.FUNDAMENTAL_DELTA_WINDOW, fill_method=None)

    def _build_turnover_proxy(self, market_data: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
        for column in (Col.TURNOVER, "turnover", Col.AMOUNT, "amount", Col.VOLUME, "volume"):
            panel = self._unstack_optional(market_data, column)
            if panel is not None:
                panel = panel.reindex(index=close.index, columns=close.columns)
                if column in (Col.TURNOVER, "turnover"):
                    return panel
                if column in (Col.AMOUNT, "amount", Col.VOLUME, "volume"):
                    # amount / volume 不是换手率，需相对自身长期均量/成交额去量纲。
                    base = panel.rolling(252, min_periods=20).mean()
                    return panel / base.replace(0, np.nan)
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    @staticmethod
    def _rolling_avg_corr(returns: pd.DataFrame, window: int) -> pd.Series:
        values = []
        min_periods = max(5, window // 3)
        for end in range(len(returns)):
            start = max(0, end - window + 1)
            sample = returns.iloc[start:end + 1].dropna(axis=1, how="all")
            if len(sample) < min_periods or sample.shape[1] < 2:
                values.append(np.nan)
                continue
            corr = sample.corr()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
            values.append(upper.mean() if not upper.empty else np.nan)
        return pd.Series(values, index=returns.index)

    @staticmethod
    def _top_bottom_spread(returns: pd.DataFrame) -> pd.Series:
        values = []
        for _, row in returns.iterrows():
            row = row.dropna().sort_values()
            if len(row) < 2:
                values.append(np.nan)
                continue
            count = min(3, max(1, len(row) // 2))
            values.append(row.tail(count).mean() - row.head(count).mean())
        return pd.Series(values, index=returns.index)

    @staticmethod
    def _group_relative_return(
        close_all: pd.DataFrame,
        left_symbols: list[str],
        right_symbols: list[str],
        window: int,
    ) -> pd.Series:
        left = [symbol for symbol in left_symbols if symbol in close_all.columns]
        right = [symbol for symbol in right_symbols if symbol in close_all.columns]
        if not left or not right:
            return pd.Series(np.nan, index=close_all.index)
        left_ret = close_all[left].pct_change(window, fill_method=None).mean(axis=1)
        right_ret = close_all[right].pct_change(window, fill_method=None).mean(axis=1)
        return left_ret - right_ret

    @staticmethod
    def _to_monthly_market_data(market_data: pd.DataFrame) -> pd.DataFrame:
        if market_data.empty:
            return market_data.copy()

        df = market_data.reset_index().copy()
        df[Col.DATE] = pd.to_datetime(df[Col.DATE]).dt.tz_localize(None)
        df = df.sort_values([Col.SYMBOL, Col.DATE])
        df["_month"] = df[Col.DATE].dt.to_period("M")

        agg_map: dict[str, str] = {}
        for column, method in (
            (Col.DATE, "last"),
            (Col.OPEN, "first"),
            (Col.HIGH, "max"),
            (Col.LOW, "min"),
            (Col.CLOSE, "last"),
            (Col.VOLUME, "sum"),
            (Col.AMOUNT, "sum"),
            (Col.ADJ_CLOSE, "last"),
        ):
            if column in df.columns:
                agg_map[column] = method

        monthly = (
            df.groupby([Col.SYMBOL, "_month"], observed=True)
            .agg(agg_map)
            .reset_index()
        )
        monthly = monthly.drop(columns="_month")
        required = [column for column in (Col.OPEN, Col.HIGH, Col.LOW, Col.CLOSE) if column in monthly.columns]
        monthly = monthly.dropna(subset=required, how="any")
        monthly = monthly.set_index([Col.DATE, Col.SYMBOL]).sort_index()
        monthly.index.names = [Col.DATE, Col.SYMBOL]
        return monthly

    def _build_monthly_market_features(
        self,
        close_all: pd.DataFrame,
        volume_all: pd.DataFrame | None,
        etf_symbols: list[object],
    ) -> pd.DataFrame:
        usable_etfs = [symbol for symbol in etf_symbols if symbol in close_all.columns]
        proxy = close_all[usable_etfs] if usable_etfs else close_all
        returns = proxy.pct_change(fill_method=None)
        market_return = returns.mean(axis=1)
        market_level = (1.0 + market_return.fillna(0.0)).cumprod()

        features = pd.DataFrame(index=close_all.index)
        for window in (1, 3, 6):
            features[f"market_ret_{window}m"] = market_level.pct_change(window, fill_method=None)
        for window in (3, 6, 12):
            features[f"market_vol_{window}m"] = (
                market_return.rolling(window, min_periods=max(2, window // 2)).std() * np.sqrt(12)
            )

        rolling_peak = market_level.rolling(12, min_periods=3).max()
        features["market_drawdown_12m"] = market_level / rolling_peak - 1.0

        for window in (1, 3):
            sector_ret = proxy.pct_change(window, fill_method=None)
            features[f"sector_dispersion_{window}m"] = sector_ret.std(axis=1)
            valid_count = sector_ret.notna().sum(axis=1)
            positive_count = sector_ret.gt(0).sum(axis=1)
            features[f"sector_breadth_{window}m"] = positive_count / valid_count.replace(0, np.nan)

        for window in (1, 3):
            features[f"growth_defensive_ret_{window}m"] = self._group_relative_return(
                close_all,
                self.GROWTH_ETFS,
                self.DEFENSIVE_ETFS,
                window,
            )
            features[f"cyclical_defensive_ret_{window}m"] = self._group_relative_return(
                close_all,
                self.CYCLICAL_ETFS,
                self.DEFENSIVE_ETFS,
                window,
            )

        if volume_all is not None:
            volume_proxy = volume_all[[symbol for symbol in usable_etfs if symbol in volume_all.columns]]
            if volume_proxy.empty:
                volume_proxy = volume_all
            total_volume = volume_proxy.sum(axis=1, min_count=1)
            vol_mean = total_volume.rolling(3, min_periods=2).mean()
            vol_std = total_volume.rolling(3, min_periods=2).std()
            features["market_volume_zscore_3m"] = (total_volume - vol_mean) / vol_std.replace(0, np.nan)

        return features.replace([np.inf, -np.inf], np.nan)

    def _ordered_model_weights(
        self,
        hist_returns: pd.DataFrame,
        hist_ranks: pd.DataFrame,
        hist_features: pd.DataFrame,
        current_features: pd.Series,
    ) -> pd.Series | None:
        feature_cols = hist_features.dropna(axis=1, how="all").columns
        if len(feature_cols) == 0:
            return None

        scores = pd.Series(np.nan, index=self.FACTOR_NAMES, dtype=float)
        rank_count = len(self.FACTOR_NAMES)
        cutoff = max(1, int(np.floor(rank_count * self.TOP_FRAC)))

        for factor_name in self.FACTOR_NAMES:
            y = hist_ranks[factor_name]
            data = hist_features[feature_cols].join(y.rename("rank")).replace([np.inf, -np.inf], np.nan)
            data = data.dropna(subset=["rank"])
            data = data[data["rank"].between(1, rank_count)]
            if len(data) < max(30, len(feature_cols) + 5) or data["rank"].nunique() < 2:
                continue

            x = data[feature_cols]
            med = x.median()
            x = x.fillna(med).dropna(axis=1, how="any")
            if x.empty:
                continue
            x_mean = x.mean()
            x_std = x.std(ddof=0).replace(0, np.nan)
            x = ((x - x_mean) / x_std).dropna(axis=1, how="any")
            if x.empty:
                continue

            y = data.loc[x.index, "rank"].astype(int)
            unique_labels = np.sort(y.unique())
            current = current_features.reindex(x.columns).fillna(med.reindex(x.columns))
            current = ((current - x_mean.reindex(x.columns)) / x_std.reindex(x.columns)).fillna(0.0)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = OrderedModel(y, x, distr="logit")
                    fitted = model.fit(method="bfgs", disp=False, maxiter=100)
                    pred = fitted.model.predict(fitted.params, exog=pd.DataFrame([current], columns=x.columns))
            except Exception:
                continue

            probs = np.asarray(pred).reshape(-1)
            model_labels = getattr(fitted.model, "labels", None)
            labels = np.asarray(model_labels, dtype=int) if model_labels is not None else unique_labels
            if len(labels) != len(probs):
                continue
            scores[factor_name] = float(probs[labels <= cutoff].sum())

        scores = scores.clip(lower=0).fillna(0.0)
        if scores.sum() <= 0:
            return None
        return scores / scores.sum()

    def _fallback_weights(
        self,
        hist_returns: pd.DataFrame,
        previous: pd.Series | None,
    ) -> pd.Series:
        mean_returns = hist_returns.reindex(columns=self.FACTOR_NAMES).mean(skipna=True)
        available = mean_returns.dropna()
        if len(available) >= self.MIN_AVAILABLE_FACTORS:
            ranks = available.rank(ascending=False, method="first")
            scores = len(available) - ranks + 1.0
            weights = pd.Series(0.0, index=self.FACTOR_NAMES)
            weights.loc[scores.index] = scores / scores.sum()
            return weights
        if previous is not None and previous.sum() > 0:
            return previous
        return self._equal_weights(self.FACTOR_NAMES)

    @staticmethod
    def _equal_weights(factor_names: list[str]) -> pd.Series:
        weights = pd.Series(1.0 / len(factor_names), index=factor_names, dtype=float)
        return weights

    @staticmethod
    def _cross_sectional_zscore(factor: pd.DataFrame) -> pd.DataFrame:
        lower = factor.quantile(0.01, axis=1)
        upper = factor.quantile(0.99, axis=1)
        clipped = factor.clip(lower=lower, upper=upper, axis=0)
        mean = clipped.mean(axis=1)
        std = clipped.std(axis=1).replace(0, np.nan)
        return clipped.sub(mean, axis=0).div(std, axis=0)


@register_factor
class OrdinalFactorRotationEnhance(OrdinalFactorRotation):
    name = "ordinal_factor_rotation_enhance"
    description = "基于 Alpha158 月频子因子的序数回归轮动"
    category = "composite"

    FACTOR_NAMES = [
        "alpha158_std60",
        "alpha158_klen",
        "alpha158_std20",
        "alpha158_std30",
        "alpha158_kup",
        "alpha158_high0",
        "alpha158_std10",
        "alpha158_std5",
        "alpha158_max5",
        "alpha158_max10",
    ]
    TRAIN_WINDOW = 60
    WEIGHT_UPDATE_INTERVAL = 1
    MIN_AVAILABLE_FACTORS = 5
    MIN_STOCK_FACTORS = 3

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        monthly_market = self._to_monthly_market_data(market_data)
        close_all = monthly_market[Col.CLOSE].unstack(Col.SYMBOL).sort_index()
        volume_all = self._unstack_optional(monthly_market, Col.VOLUME)
        if volume_all is not None:
            volume_all = volume_all.reindex(index=close_all.index, columns=close_all.columns)

        all_symbols = list(close_all.columns)
        etf_symbols = [symbol for symbol in all_symbols if self._is_etf_symbol(str(symbol))]
        stock_symbols = [symbol for symbol in all_symbols if symbol not in etf_symbols]
        if not stock_symbols:
            raise ValueError("ordinal_factor_rotation_enhance 无可用股票标的")

        close = close_all[stock_symbols]
        factor_dict = self._compute_alpha158_sub_factors(monthly_market, close)
        available = [name for name, factor in factor_dict.items() if not factor.dropna(how="all").empty]
        if len(available) < self.MIN_AVAILABLE_FACTORS:
            raise ValueError(
                f"ordinal_factor_rotation_enhance 可用子因子不足: {len(available)} < {self.MIN_AVAILABLE_FACTORS}"
            )

        stock_returns_forward = close.pct_change(fill_method=None).shift(-1)
        market_features = self._build_monthly_market_features(close_all, volume_all, etf_symbols)
        factor_returns = self._compute_factor_returns(
            factor_dict,
            stock_returns_forward,
            top_quantile=self.TOP_QUANTILE,
        )
        factor_ranks = self._factor_returns_to_ranks(factor_returns)
        factor_weights = self._compute_dynamic_factor_weights(
            factor_returns,
            factor_ranks,
            market_features,
        )
        signals = self._combine_factor_signals(factor_dict, factor_weights)
        signals = signals.reindex(index=close.index, columns=stock_symbols)
        signals.index.name = Col.DATE
        signals.columns.name = Col.SYMBOL

        self.last_monthly_market_data = monthly_market
        self.last_market_features = market_features
        self.last_factor_dict = factor_dict
        self.last_factor_returns = factor_returns
        self.last_factor_ranks = factor_ranks
        self.last_factor_weights = factor_weights
        return signals

    def _compute_alpha158_sub_factors(
        self,
        monthly_market: pd.DataFrame,
        close: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        blank = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        factor_classes = _alpha158_factor_class_map()
        factors: dict[str, pd.DataFrame] = {}
        for factor_name in self.FACTOR_NAMES:
            factor_cls = factor_classes.get(factor_name)
            if factor_cls is None:
                factors[factor_name] = blank.copy()
                continue

            factor = factor_cls().generate_signals(monthly_market, None)
            if not isinstance(factor, pd.DataFrame) or factor.empty:
                factors[factor_name] = blank.copy()
                continue
            factor = factor.apply(pd.to_numeric, errors="coerce")
            factors[factor_name] = factor.reindex(index=close.index, columns=close.columns)
        return factors
