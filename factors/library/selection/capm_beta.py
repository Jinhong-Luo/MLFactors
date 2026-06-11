"""CAPM Beta stock-selection factor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.schema import Col
from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class CapmBeta(BaseFactor):
    name = "capm_beta"
    description = "CAPM beta computed from rolling stock and market daily returns"
    category = "risk"

    def __init__(
        self,
        *,
        lookback: int = 252,
        min_obs: int = 120,
        clip: tuple[float, float] | None = (-3.0, 3.0),
        rebalance: str = "monthly",
    ) -> None:
        if rebalance not in {"daily", "monthly"}:
            raise ValueError("rebalance must be either 'daily' or 'monthly'")
        self.lookback = lookback
        self.min_obs = min_obs
        self.clip = clip
        self.rebalance = rebalance

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        market_data = data["market"].sort_index()
        fundamental = data["fundamental"].sort_index()

        close = market_data[Col.CLOSE].unstack(Col.SYMBOL).astype(float).sort_index()
        close.index.name = Col.DATE
        close.columns.name = Col.SYMBOL
        stock_ret = close.pct_change(fill_method=None)

        market_cap = fundamental["market_cap"].unstack(Col.SYMBOL).astype(float).sort_index()
        market_cap = market_cap.reindex(index=stock_ret.index, columns=stock_ret.columns).ffill()
        weights = market_cap.where(market_cap > 0)
        weights = weights.where(stock_ret.notna())
        weights = weights.div(weights.sum(axis=1), axis=0)
        market_ret = (stock_ret * weights).sum(axis=1, min_count=1)

        if self.min_obs > self.lookback:
            beta = pd.DataFrame(np.nan, index=stock_ret.index, columns=stock_ret.columns)
        else:
            rolling_cov = stock_ret.rolling(
                self.lookback,
                min_periods=self.min_obs,
            ).cov(market_ret)
            rolling_var = market_ret.rolling(
                self.lookback,
                min_periods=self.min_obs,
            ).var()
            beta = rolling_cov.div(rolling_var.where(rolling_var != 0), axis=0)
        beta = beta.reindex(index=stock_ret.index, columns=stock_ret.columns)
        beta = beta.replace([np.inf, -np.inf], np.nan)
        if self.clip is not None:
            beta = beta.clip(lower=self.clip[0], upper=self.clip[1])

        if "signal_dates" in data:
            signal_dates = pd.DatetimeIndex(pd.to_datetime(data["signal_dates"])).sort_values()
            beta = beta.reindex(index=signal_dates)
        elif self.rebalance == "monthly":
            beta = beta.loc[~beta.index.to_period("M").duplicated(keep="last")]

        beta.index.name = Col.DATE
        beta.columns.name = Col.SYMBOL
        return beta
