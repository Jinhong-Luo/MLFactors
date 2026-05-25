"""Alpha158 滚动窗口因子。"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from data.schema import Col
from factors.base import BaseFactor
from factors.registry import register_factor

WINDOWS: tuple[int, ...] = (5, 10, 20, 30, 60)
PREFIXES: tuple[str, ...] = (
    "ROC",
    "MA",
    "STD",
    "BETA",
    "RSQR",
    "RESI",
    "MAX",
    "MIN",
    "QTLU",
    "QTLD",
    "RANK",
    "RSV",
    "IMAX",
    "IMIN",
    "IMXD",
    "CORR",
    "CORD",
    "CNTP",
    "CNTN",
    "CNTD",
    "SUMP",
    "SUMN",
    "SUMD",
    "VMA",
    "VSTD",
    "WVMA",
    "VSUMP",
    "VSUMN",
    "VSUMD",
)

_DESCRIPTION_MAP: dict[str, str] = {
    "ROC": "过去 N 日收盘价相对当前收盘价",
    "MA": "N 日均价相对当前收盘价",
    "STD": "N 日收盘价标准差相对当前收盘价",
    "BETA": "N 日收盘价线性回归斜率相对当前收盘价",
    "RSQR": "N 日收盘价线性回归 R 方",
    "RESI": "N 日收盘价线性回归末端残差相对当前收盘价",
    "MAX": "N 日最高价相对当前收盘价",
    "MIN": "N 日最低价相对当前收盘价",
    "QTLU": "N 日收盘价 80% 分位数相对当前收盘价",
    "QTLD": "N 日收盘价 20% 分位数相对当前收盘价",
    "RANK": "当前收盘价在过去 N 日收盘价中的滚动百分位",
    "RSV": "当前收盘价在过去 N 日高低区间中的位置",
    "IMAX": "N 日最高价所在位置的相对索引",
    "IMIN": "N 日最低价所在位置的相对索引",
    "IMXD": "N 日最高价位置与最低价位置之差",
    "CORR": "N 日收盘价与对数成交量的相关性",
    "CORD": "N 日价格变化率与对数成交量变化率的相关性",
    "CNTP": "N 日上涨天数占比",
    "CNTN": "N 日下跌天数占比",
    "CNTD": "上涨天数占比减下跌天数占比",
    "SUMP": "N 日正价格变化占总波动比例",
    "SUMN": "N 日负价格变化占总波动比例",
    "SUMD": "正价格变化占比减负价格变化占比",
    "VMA": "N 日平均成交量相对当前成交量",
    "VSTD": "N 日成交量标准差相对当前成交量",
    "WVMA": "N 日成交量加权绝对收益的波动率相对均值",
    "VSUMP": "N 日成交量上升幅度占总成交量波动比例",
    "VSUMN": "N 日成交量下降幅度占总成交量波动比例",
    "VSUMD": "成交量上升幅度占比减成交量下降幅度占比",
}


def _extract_panel(market_data: pd.DataFrame, column: str) -> pd.DataFrame:
    panel = market_data[column].astype(float).unstack(Col.SYMBOL).sort_index()
    panel.index.name = Col.DATE
    panel.columns.name = Col.SYMBOL
    return panel


def _extract_inputs(
    market_data: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        "close": _extract_panel(market_data, Col.CLOSE),
        "high": _extract_panel(market_data, Col.HIGH),
        "low": _extract_panel(market_data, Col.LOW),
        "volume": _extract_panel(market_data, Col.VOLUME),
    }


def _finalize(signals: pd.DataFrame) -> pd.DataFrame:
    signals = signals.replace([np.inf, -np.inf], np.nan)
    signals.index.name = Col.DATE
    signals.columns.name = Col.SYMBOL
    return signals


def _safe_divide(
    numerator: pd.DataFrame,
    denominator: pd.DataFrame | float,
) -> pd.DataFrame:
    if isinstance(denominator, pd.DataFrame):
        safe_denominator = denominator.where(denominator != 0)
    else:
        safe_denominator = np.nan if denominator == 0 else denominator
    return _finalize(numerator.divide(safe_denominator))


def _rolling_apply(
    frame: pd.DataFrame,
    window: int,
    func: Callable[[np.ndarray], float],
) -> pd.DataFrame:
    return _finalize(frame.rolling(window=window, min_periods=window).apply(func, raw=True))


def _safe_log_volume(volume: pd.DataFrame) -> pd.DataFrame:
    positive_volume = volume.where(volume > 0)
    return _finalize(np.log(positive_volume))


def _rolling_regression(values: np.ndarray) -> tuple[float, float, float]:
    if not np.isfinite(values).all():
        return np.nan, np.nan, np.nan

    x = np.arange(values.size, dtype=float)
    y = values.astype(float, copy=False)

    y_mean = y.mean()
    y_centered = y - y_mean
    ss_tot = float(np.dot(y_centered, y_centered))

    if ss_tot == 0.0:
        return 0.0, 1.0, 0.0

    x_centered = x - x.mean()
    ss_x = float(np.dot(x_centered, x_centered))
    if ss_x == 0.0:
        return np.nan, np.nan, np.nan

    slope = float(np.dot(x_centered, y_centered) / ss_x)
    intercept = float(y_mean - slope * x.mean())
    fitted = intercept + slope * x
    residuals = y - fitted
    ss_res = float(np.dot(residuals, residuals))
    rsqr = 1.0 - ss_res / ss_tot
    return slope, rsqr, float(residuals[-1])


def _rolling_beta(close: pd.DataFrame, window: int) -> pd.DataFrame:
    beta = _rolling_apply(close, window, lambda values: _rolling_regression(values)[0])
    return _safe_divide(beta, close)


def _rolling_rsqr(close: pd.DataFrame, window: int) -> pd.DataFrame:
    return _rolling_apply(close, window, lambda values: _rolling_regression(values)[1])


def _rolling_resi(close: pd.DataFrame, window: int) -> pd.DataFrame:
    residual = _rolling_apply(close, window, lambda values: _rolling_regression(values)[2])
    return _safe_divide(residual, close)


def _rolling_rank(close: pd.DataFrame, window: int) -> pd.DataFrame:
    return _rolling_apply(
        close,
        window,
        lambda values: float(pd.Series(values).rank(pct=True).iloc[-1]),
    )


def _rolling_index_of_extreme(
    frame: pd.DataFrame,
    window: int,
    reducer: Callable[[np.ndarray], int],
) -> pd.DataFrame:
    positions = _rolling_apply(frame, window, lambda values: float(reducer(values)))
    return _safe_divide(positions, float(window - 1))


def _rolling_corr(left: pd.DataFrame, right: pd.DataFrame, window: int) -> pd.DataFrame:
    return _finalize(left.rolling(window=window, min_periods=window).corr(right))


def _positive_part(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.clip(lower=0)


def _negative_part(frame: pd.DataFrame) -> pd.DataFrame:
    return (-frame).clip(lower=0)


def _rolling_ratio(
    numerator: pd.DataFrame,
    denominator: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    numer = numerator.rolling(window=window, min_periods=window).sum()
    denom = denominator.rolling(window=window, min_periods=window).sum()
    return _safe_divide(numer, denom)


def _compute_rolling_feature(
    market_data: pd.DataFrame,
    prefix: str,
    window: int,
) -> pd.DataFrame:
    panels = _extract_inputs(market_data)
    close = panels["close"]
    high = panels["high"]
    low = panels["low"]
    volume = panels["volume"]

    close_return = close.pct_change(fill_method=None)
    close_delta = close.diff()
    volume_delta = volume.diff()
    log_volume = _safe_log_volume(volume)
    log_volume_delta = log_volume.diff()

    if prefix == "ROC":
        return _safe_divide(close.shift(window), close)
    if prefix == "MA":
        return _safe_divide(close.rolling(window=window, min_periods=window).mean(), close)
    if prefix == "STD":
        return _safe_divide(close.rolling(window=window, min_periods=window).std(), close)
    if prefix == "BETA":
        return _rolling_beta(close, window)
    if prefix == "RSQR":
        return _rolling_rsqr(close, window)
    if prefix == "RESI":
        return _rolling_resi(close, window)
    if prefix == "MAX":
        return _safe_divide(high.rolling(window=window, min_periods=window).max(), close)
    if prefix == "MIN":
        return _safe_divide(low.rolling(window=window, min_periods=window).min(), close)
    if prefix == "QTLU":
        return _safe_divide(close.rolling(window=window, min_periods=window).quantile(0.8), close)
    if prefix == "QTLD":
        return _safe_divide(close.rolling(window=window, min_periods=window).quantile(0.2), close)
    if prefix == "RANK":
        return _rolling_rank(close, window)
    if prefix == "RSV":
        rolling_low = low.rolling(window=window, min_periods=window).min()
        rolling_high = high.rolling(window=window, min_periods=window).max()
        return _safe_divide(close - rolling_low, rolling_high - rolling_low)
    if prefix == "IMAX":
        return _rolling_index_of_extreme(high, window, np.argmax)
    if prefix == "IMIN":
        return _rolling_index_of_extreme(low, window, np.argmin)
    if prefix == "IMXD":
        imax = _rolling_apply(high, window, lambda values: float(np.argmax(values)))
        imin = _rolling_apply(low, window, lambda values: float(np.argmin(values)))
        return _safe_divide(imax - imin, float(window - 1))
    if prefix == "CORR":
        return _rolling_corr(close, log_volume, window)
    if prefix == "CORD":
        return _rolling_corr(close_return, log_volume_delta, window)
    if prefix == "CNTP":
        up = close_delta.gt(0).where(close_delta.notna())
        return _finalize(up.rolling(window=window, min_periods=window).mean())
    if prefix == "CNTN":
        down = close_delta.lt(0).where(close_delta.notna())
        return _finalize(down.rolling(window=window, min_periods=window).mean())
    if prefix == "CNTD":
        up = close_delta.gt(0).where(close_delta.notna())
        down = close_delta.lt(0).where(close_delta.notna())
        return _finalize(
            up.rolling(window=window, min_periods=window).mean()
            - down.rolling(window=window, min_periods=window).mean()
        )
    if prefix == "SUMP":
        return _rolling_ratio(_positive_part(close_delta), close_delta.abs(), window)
    if prefix == "SUMN":
        return _rolling_ratio(_negative_part(close_delta), close_delta.abs(), window)
    if prefix == "SUMD":
        sump = _rolling_ratio(_positive_part(close_delta), close_delta.abs(), window)
        sumn = _rolling_ratio(_negative_part(close_delta), close_delta.abs(), window)
        return _finalize(sump - sumn)
    if prefix == "VMA":
        return _safe_divide(volume.rolling(window=window, min_periods=window).mean(), volume)
    if prefix == "VSTD":
        return _safe_divide(volume.rolling(window=window, min_periods=window).std(), volume)
    if prefix == "WVMA":
        weighted_move = close_return.abs() * volume
        weighted_std = weighted_move.rolling(window=window, min_periods=window).std()
        weighted_mean = weighted_move.rolling(window=window, min_periods=window).mean()
        return _safe_divide(weighted_std, weighted_mean)
    if prefix == "VSUMP":
        return _rolling_ratio(_positive_part(volume_delta), volume_delta.abs(), window)
    if prefix == "VSUMN":
        return _rolling_ratio(_negative_part(volume_delta), volume_delta.abs(), window)
    if prefix == "VSUMD":
        vsump = _rolling_ratio(_positive_part(volume_delta), volume_delta.abs(), window)
        vsumn = _rolling_ratio(_negative_part(volume_delta), volume_delta.abs(), window)
        return _finalize(vsump - vsumn)

    raise ValueError(f"未知的 Alpha158 rolling 前缀: {prefix}")


class _Alpha158RollingFactor(BaseFactor):
    name = ""
    description = ""
    category = "alpha158"
    prefix = ""
    window = 0

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return _compute_rolling_feature(market_data, self.prefix, self.window)


def _make_factor_class(prefix: str, window: int) -> type[BaseFactor]:
    class_name = f"Alpha158{prefix.title()}{window}"
    factor_name = f"alpha158_{prefix.lower()}{window}"
    description = f"Alpha158 {prefix}{window}，{_DESCRIPTION_MAP[prefix]}"
    attrs = {
        "__module__": __name__,
        "name": factor_name,
        "description": description,
        "category": "alpha158",
        "prefix": prefix,
        "window": window,
    }
    factor_cls = type(class_name, (_Alpha158RollingFactor,), attrs)
    globals()[class_name] = factor_cls
    return register_factor(factor_cls)


for _prefix in PREFIXES:
    for _window in WINDOWS:
        _make_factor_class(_prefix, _window)

