"""CAPM Beta 因子。"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class Beta(BaseFactor):
    """使用市值加权市场组合收益计算滚动 Beta。"""

    name = "beta"
    description = "CAPM Beta"

    def __init__(
        self,
        lookback: int = 252,
        min_obs: int = 120,
        clip: tuple[float, float] | None = (-3.0, 3.0),
    ) -> None:
        self.lookback = lookback
        self.min_obs = min_obs
        self.clip = clip

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        close = data["close"].unstack("symbol").astype(float).sort_index()
        market_cap = data["market_cap"].unstack("symbol").astype(float).sort_index()
        market_cap = market_cap.reindex(index=close.index, columns=close.columns)

        stock_ret = close.pct_change(fill_method=None)
        weights = market_cap.where(market_cap > 0).where(stock_ret.notna())
        constituent_mask = None

        if constituents:
            constituent_mask = pd.DataFrame(
                [
                    [symbol in constituents.get(str(date.date()), set()) for symbol in close.columns]
                    for date in close.index
                ],
                index=close.index,
                columns=close.columns,
            )
            weights = weights.where(constituent_mask)

        weights = weights.div(weights.sum(axis=1), axis=0)
        market_ret = (stock_ret * weights).sum(axis=1, min_count=1)

        if self.min_obs > self.lookback:
            beta = pd.DataFrame(np.nan, index=stock_ret.index, columns=stock_ret.columns)
        else:
            market_ret_frame = pd.DataFrame(
                np.repeat(market_ret.to_numpy()[:, None], len(stock_ret.columns), axis=1),
                index=stock_ret.index,
                columns=stock_ret.columns,
            )
            valid = stock_ret.notna() & market_ret_frame.notna()
            stock_ret = stock_ret.where(valid)
            market_ret_frame = market_ret_frame.where(valid)

            market_var = market_ret_frame.rolling(
                self.lookback,
                min_periods=self.min_obs,
            ).var()
            beta = stock_ret.rolling(
                self.lookback,
                min_periods=self.min_obs,
            ).cov(market_ret_frame)
            beta = beta.div(market_var.where(market_var != 0))

        beta = beta.replace([np.inf, -np.inf], np.nan)
        if self.clip is not None:
            beta = beta.clip(lower=self.clip[0], upper=self.clip[1])
        if constituent_mask is not None:
            beta = beta.where(constituent_mask)

        beta.index.name = "date"
        beta.columns.name = "symbol"
        return beta
