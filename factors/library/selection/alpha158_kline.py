"""Alpha158 K 线形态因子。"""

from __future__ import annotations

import pandas as pd

from data.schema import Col
from factors.base import BaseFactor
from factors.registry import register_factor


def _extract_price_panels(
    market_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    open_ = market_data[Col.OPEN].unstack(Col.SYMBOL)
    high = market_data[Col.HIGH].unstack(Col.SYMBOL)
    low = market_data[Col.LOW].unstack(Col.SYMBOL)
    close = market_data[Col.CLOSE].unstack(Col.SYMBOL)
    return open_, high, low, close


def _safe_divide(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    safe_denominator = denominator.where(denominator != 0)
    signals = numerator.divide(safe_denominator)
    signals.index.name = Col.DATE
    signals.columns.name = Col.SYMBOL
    return signals


def _upper_body_edge(open_: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    return open_.where(open_ >= close, close)


def _lower_body_edge(open_: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    return open_.where(open_ <= close, close)


def _intraday_range(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    return high - low


@register_factor
class Alpha158Kmid(BaseFactor):
    name = "alpha158_kmid"
    description = "Alpha158 KMID，实体相对开盘价"
    category = "alpha158"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        open_, _, _, close = _extract_price_panels(market_data)
        return _safe_divide(close - open_, open_)


@register_factor
class Alpha158Klen(BaseFactor):
    name = "alpha158_klen"
    description = "Alpha158 KLEN，K 线长度相对开盘价"
    category = "alpha158"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        open_, high, low, _ = _extract_price_panels(market_data)
        return _safe_divide(_intraday_range(high, low), open_)


@register_factor
class Alpha158Kmid2(BaseFactor):
    name = "alpha158_kmid2"
    description = "Alpha158 KMID2，实体相对全日振幅"
    category = "alpha158"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        open_, high, low, close = _extract_price_panels(market_data)
        return _safe_divide(close - open_, _intraday_range(high, low))


@register_factor
class Alpha158Kup(BaseFactor):
    name = "alpha158_kup"
    description = "Alpha158 KUP，上影线相对开盘价"
    category = "alpha158"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        open_, high, _, close = _extract_price_panels(market_data)
        upper_shadow = high - _upper_body_edge(open_, close)
        return _safe_divide(upper_shadow, open_)


@register_factor
class Alpha158Kup2(BaseFactor):
    name = "alpha158_kup2"
    description = "Alpha158 KUP2，上影线相对全日振幅"
    category = "alpha158"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        open_, high, low, close = _extract_price_panels(market_data)
        upper_shadow = high - _upper_body_edge(open_, close)
        return _safe_divide(upper_shadow, _intraday_range(high, low))


@register_factor
class Alpha158Klow(BaseFactor):
    name = "alpha158_klow"
    description = "Alpha158 KLOW，下影线相对开盘价"
    category = "alpha158"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        open_, _, low, close = _extract_price_panels(market_data)
        lower_shadow = _lower_body_edge(open_, close) - low
        return _safe_divide(lower_shadow, open_)


@register_factor
class Alpha158Klow2(BaseFactor):
    name = "alpha158_klow2"
    description = "Alpha158 KLOW2，下影线相对全日振幅"
    category = "alpha158"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        open_, high, low, close = _extract_price_panels(market_data)
        lower_shadow = _lower_body_edge(open_, close) - low
        return _safe_divide(lower_shadow, _intraday_range(high, low))


@register_factor
class Alpha158Ksft(BaseFactor):
    name = "alpha158_ksft"
    description = "Alpha158 KSFT，收盘位置偏移相对开盘价"
    category = "alpha158"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        open_, high, low, close = _extract_price_panels(market_data)
        return _safe_divide(2 * close - high - low, open_)


@register_factor
class Alpha158Ksft2(BaseFactor):
    name = "alpha158_ksft2"
    description = "Alpha158 KSFT2，收盘位置偏移相对全日振幅"
    category = "alpha158"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        open_, high, low, close = _extract_price_panels(market_data)
        return _safe_divide(2 * close - high - low, _intraday_range(high, low))
