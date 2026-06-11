"""SVM Alpha158 stock-selection score."""

from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.schema import Col
from factors.base import BaseFactor
from factors.registry import register_factor
from lab.SVM_research import get_month_end_dates, preprocess_features, predict_svm_scores


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = ROOT / "outputs" / "svm_research_2017_2023_nona"
MARKET_CAP_COLUMNS = ("market_cap", Col.MKT_CAP, "total_mv", "circ_mv")


def set_artifact_dir(path: str | Path) -> None:
    global ARTIFACT_DIR
    ARTIFACT_DIR = Path(path).expanduser().resolve()
    _load_artifacts.cache_clear()


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[Any, Any, dict[str, float], list[str]]:
    with (ARTIFACT_DIR / "svm_model.pkl").open("rb") as f:
        model = pickle.load(f)
    with (ARTIFACT_DIR / "pca.pkl").open("rb") as f:
        pca = pickle.load(f)
    with (ARTIFACT_DIR / "best_params.json").open("r", encoding="utf-8") as f:
        best_params = json.load(f)
    with (ARTIFACT_DIR / "feature_columns.json").open("r", encoding="utf-8") as f:
        feature_columns = json.load(f)
    return model, pca, best_params, feature_columns


def _normalize_panel_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("SVM Alpha158 因子需要 MultiIndex[date, symbol] 数据")
    names = list(df.index.names)
    date_level = names.index(Col.DATE) if Col.DATE in names else 0
    symbol_level = names.index(Col.SYMBOL) if Col.SYMBOL in names else 1
    dates = pd.DatetimeIndex(pd.to_datetime(df.index.get_level_values(date_level))).tz_localize(None)
    symbols = df.index.get_level_values(symbol_level).astype(str)
    normalized = df.copy()
    normalized.index = pd.MultiIndex.from_arrays([dates, symbols], names=[Col.DATE, Col.SYMBOL])
    return normalized.sort_index()


def _resolve_signal_dates(data: dict[str, pd.DataFrame], alpha_panel: pd.DataFrame) -> pd.DatetimeIndex:
    if data.get("signal_dates") is not None:
        dates = pd.DatetimeIndex(pd.to_datetime(data["signal_dates"])).tz_localize(None)
    elif data.get("market") is not None and not data["market"].empty:
        market_dates = data["market"].index.get_level_values(Col.DATE)
        dates = get_month_end_dates(market_dates)
    else:
        dates = pd.DatetimeIndex(alpha_panel.index.get_level_values(Col.DATE).unique())
    return dates.unique().sort_values()


def _market_cap_panel(
    data: dict[str, pd.DataFrame],
    signal_dates: pd.DatetimeIndex,
    symbols: pd.Index,
) -> pd.DataFrame:
    for table_name in ("fundamental", "market"):
        source = data.get(table_name)
        if source is None or source.empty:
            continue
        source = _normalize_panel_index(source)
        for column in MARKET_CAP_COLUMNS:
            if column in source.columns:
                cap = source[column].unstack(Col.SYMBOL).sort_index()
                aligned = cap.reindex(signal_dates).reindex(columns=symbols)
                medians = cap.reindex(signal_dates).median(axis=1, skipna=True)
                return aligned.T.fillna(medians).T
    raise ValueError("SVM Alpha158 因子需要市值列: market_cap / mkt_cap / total_mv / circ_mv")


def _valid_feature_dates(X_raw: pd.DataFrame) -> pd.DatetimeIndex:
    clean = X_raw.replace([np.inf, -np.inf], np.nan)
    dates = clean.index.get_level_values(Col.DATE).unique().sort_values()
    valid_dates = []
    for date in dates:
        block = clean.loc[clean.index.get_level_values(Col.DATE) == date]
        if block.notna().to_numpy().all():
            valid_dates.append(date)
    return pd.DatetimeIndex(valid_dates)


@register_factor
class SVMAlpha158(BaseFactor):
    name = "svm_alpha158"
    description = "RBF SVM score using precomputed Alpha158 exposures and nona PCA95 artifacts"
    category = "alpha158"
    best_params = {"C": 1.0, "gamma": 0.1}

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        alpha_panel = data.get("alpha158")
        if alpha_panel is None or alpha_panel.empty:
            raise ValueError("SVM Alpha158 因子需要 data['alpha158'] 提供预计算 Alpha158 暴露")

        alpha_panel = _normalize_panel_index(alpha_panel)
        factor_cols = [column for column in alpha_panel.columns if column.startswith("alpha158_")]
        if not factor_cols:
            raise ValueError("data['alpha158'] 中未找到 alpha158_ 因子列")

        signal_dates = _resolve_signal_dates(data, alpha_panel)
        mask = alpha_panel.index.get_level_values(Col.DATE).isin(signal_dates)
        X_raw = alpha_panel.loc[mask, factor_cols].sort_index()
        if X_raw.empty:
            raise ValueError("SVM Alpha158 月末特征面板为空，请检查 signal_dates 与 alpha158 日期")

        symbols = X_raw.index.get_level_values(Col.SYMBOL).unique()
        signals = pd.DataFrame(np.nan, index=signal_dates, columns=symbols, dtype=float)
        signals.index.name = Col.DATE
        signals.columns.name = Col.SYMBOL

        valid_dates = _valid_feature_dates(X_raw)
        if valid_dates.empty:
            return signals

        X_ready = X_raw.loc[X_raw.index.get_level_values(Col.DATE).isin(valid_dates)]
        market_cap = _market_cap_panel(data, valid_dates, symbols)
        X_processed, _ = preprocess_features(X_ready, market_cap, use_pca=True, pca_n_components=0.95)

        model, pca, _, feature_columns = _load_artifacts()
        if pca is not None:
            values = pca.transform(X_processed.replace([np.inf, -np.inf], np.nan).fillna(0.0))
            X_model = pd.DataFrame(values, index=X_processed.index, columns=feature_columns)
        else:
            X_model = X_processed

        scores = predict_svm_scores(model, X_model)
        scored = scores.unstack(Col.SYMBOL).sort_index()
        signals.loc[scored.index, scored.columns] = scored
        signals.index.name = Col.DATE
        signals.columns.name = Col.SYMBOL
        return signals
