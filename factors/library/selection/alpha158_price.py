"""Alpha158 当日价格特征。"""

from __future__ import annotations

import pandas as pd

from data.schema import Col
from factors.base import BaseFactor
from factors.registry import register_factor


def _unstack_panel(market_data: pd.DataFrame, column: str) -> pd.DataFrame:
    panel = market_data[column].unstack(Col.SYMBOL)
    panel.index.name = Col.DATE
    panel.columns.name = Col.SYMBOL
    return panel


def _safe_normalize(numerator: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    safe_close = close.where(close != 0)
    signals = numerator.divide(safe_close)
    signals.index.name = Col.DATE
    signals.columns.name = Col.SYMBOL
    return signals


def _generate_price_ratio(market_data: pd.DataFrame, price_col: str) -> pd.DataFrame:
    close = _unstack_panel(market_data, Col.CLOSE)
    price = _unstack_panel(market_data, price_col)
    return _safe_normalize(price, close)


def _resolve_vwap_panel(market_data: pd.DataFrame) -> pd.DataFrame:
    close = _unstack_panel(market_data, Col.CLOSE)
    open_ = _unstack_panel(market_data, Col.OPEN)
    high = _unstack_panel(market_data, Col.HIGH)
    low = _unstack_panel(market_data, Col.LOW)
    vwap = (
        _unstack_panel(market_data, Col.VWAP)
        if Col.VWAP in market_data.columns
        else pd.DataFrame(float("nan"), index=close.index, columns=close.columns)
    )

    fallback_to_typical = vwap.isna()
    if Col.AMOUNT in market_data.columns and Col.VOLUME in market_data.columns:
        amount = _unstack_panel(market_data, Col.AMOUNT)
        volume = _unstack_panel(market_data, Col.VOLUME)
        safe_volume = volume.where(volume != 0)
        amount_based_vwap = amount.divide(safe_volume)
        candidate_ratio = amount_based_vwap.divide(close.where(close != 0))
        plausible = candidate_ratio.ge(0.5) & candidate_ratio.le(1.5)
        amount_based_vwap = amount_based_vwap.where(plausible)

        vwap = vwap.where(vwap.notna(), amount_based_vwap)
        fallback_to_typical = vwap.isna()

    typical_price = (open_ + high + low + close) / 4.0
    vwap = vwap.where(~fallback_to_typical, typical_price)
    vwap.index.name = Col.DATE
    vwap.columns.name = Col.SYMBOL
    return vwap


@register_factor
class Alpha158Open0(BaseFactor):
    name = "alpha158_open0"
    description = "Alpha158 OPEN0: open / close"
    category = "alpha158"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return _generate_price_ratio(market_data, Col.OPEN)


@register_factor
class Alpha158High0(BaseFactor):
    name = "alpha158_high0"
    description = "Alpha158 HIGH0: high / close"
    category = "alpha158"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return _generate_price_ratio(market_data, Col.HIGH)


@register_factor
class Alpha158Low0(BaseFactor):
    name = "alpha158_low0"
    description = "Alpha158 LOW0: low / close"
    category = "alpha158"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return _generate_price_ratio(market_data, Col.LOW)


@register_factor
class Alpha158Vwap0(BaseFactor):
    name = "alpha158_vwap0"
    description = "Alpha158 VWAP0: vwap / close with fallback"
    category = "alpha158"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        close = _unstack_panel(market_data, Col.CLOSE)
        vwap = _resolve_vwap_panel(market_data)
        return _safe_normalize(vwap, close)
