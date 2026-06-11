"""Composite quality growth stock-selection factor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.schema import Col
from factors.base import BaseFactor
from factors.registry import register_factor


SUBFACTOR_NAMES = (
    "ROEQ",
    "ROAQ",
    "GPOAQ",
    "GMARQ",
    "dROEQ",
    "dROAQ",
    "dGPOAQ",
    "dGMARQ",
    "DividendRatioTTM",
)
BASE_PROFITABILITY_NAMES = ("ROEQ", "ROAQ", "GPOAQ", "GMARQ")
GROWTH_QUALITY_NAMES = ("dROEQ", "dROAQ", "dGPOAQ", "dGMARQ")

MARKET_CAP_COLUMNS = ("market_cap", Col.MKT_CAP, "total_mv", "circ_mv")
INDUSTRY_COLUMNS = ("industry", "sector", "gics_sector", "gics_industry", "sw_industry", "industry_code")
DIVIDEND_COLUMNS = ("dividend", "dividend_amount", "cash_dividend", "amount", "dividends_paid", "pay_div")


def _normalize_panel_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("quality_growth factor requires MultiIndex[date, symbol] data")
    names = list(df.index.names)
    date_level = names.index(Col.DATE) if Col.DATE in names else 0
    symbol_level = names.index(Col.SYMBOL) if Col.SYMBOL in names else 1
    dates = pd.DatetimeIndex(pd.to_datetime(df.index.get_level_values(date_level))).tz_localize(None)
    symbols = df.index.get_level_values(symbol_level).astype(str)
    result = df.copy()
    result.index = pd.MultiIndex.from_arrays([dates, symbols], names=[Col.DATE, Col.SYMBOL])
    return result.sort_index()


def _resolve_signal_dates(data: dict[str, pd.DataFrame], market_data: pd.DataFrame) -> pd.DatetimeIndex:
    signal_dates = data.get("signal_dates")
    if signal_dates is None:
        dates = market_data.index.get_level_values(Col.DATE).unique()
    elif isinstance(signal_dates, pd.DataFrame):
        dates = signal_dates[Col.DATE] if Col.DATE in signal_dates.columns else signal_dates.index
    elif isinstance(signal_dates, pd.Series):
        dates = signal_dates
    else:
        dates = signal_dates
    return pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(None).unique().sort_values()


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator.abs() > 1e-12)
    return numerator.divide(denominator)


def _first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _numeric_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    column = _first_existing_column(df, candidates)
    if column is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _prepare_statement_records(statement: pd.DataFrame) -> pd.DataFrame:
    if statement is None or statement.empty:
        raise ValueError("final_quality_growth requires data['statement'] financial report data")

    stmt = _normalize_panel_index(statement).reset_index()
    stmt[Col.DATE] = pd.to_datetime(stmt[Col.DATE]).dt.tz_localize(None)
    if "report_date" in stmt.columns:
        stmt["report_date"] = pd.to_datetime(stmt["report_date"], errors="coerce").dt.tz_localize(None)
    else:
        stmt["report_date"] = stmt[Col.DATE]
    stmt["report_date"] = stmt["report_date"].fillna(stmt[Col.DATE])
    stmt["statement_date"] = stmt[Col.DATE]
    stmt[Col.SYMBOL] = stmt[Col.SYMBOL].astype(str)
    if "fiscal_quarter" in stmt.columns:
        quarter = pd.to_numeric(stmt["fiscal_quarter"], errors="coerce")
        stmt = stmt.loc[quarter.ne(0) | quarter.isna()].copy()

    stmt["net_income_common"] = _numeric_column(stmt, ("net_income_common_stock", "net_profit", "net_income"))
    stmt["net_income_total"] = _numeric_column(stmt, ("net_income", "net_profit"))
    stmt["revenue_q"] = _numeric_column(stmt, ("revenue", "total_revenue"))
    stmt["cost_revenue_q"] = _numeric_column(stmt, ("cost_revenue", "operating_cost", "cost"))
    stmt["gross_profit_q"] = _numeric_column(stmt, ("gross_profit",))
    stmt["gross_profit_q"] = stmt["gross_profit_q"].where(
        stmt["gross_profit_q"].notna(),
        stmt["revenue_q"] - stmt["cost_revenue_q"],
    )
    stmt["total_assets_q"] = _numeric_column(stmt, ("total_assets",))
    stmt["equity_q"] = _numeric_column(stmt, ("shareholder_equity", "equity"))
    stmt = stmt.dropna(subset=[Col.DATE, Col.SYMBOL])
    stmt = stmt.sort_values([Col.SYMBOL, Col.DATE, "report_date"])
    stmt = stmt.drop_duplicates([Col.SYMBOL, Col.DATE], keep="last")

    grouped = stmt.groupby(Col.SYMBOL, group_keys=False)
    stmt["avg_assets_q"] = (stmt["total_assets_q"] + grouped["total_assets_q"].shift(1)) / 2.0
    stmt["net_income_common_yoy"] = grouped["net_income_common"].shift(4)
    stmt["net_income_total_yoy"] = grouped["net_income_total"].shift(4)
    stmt["gross_profit_yoy"] = grouped["gross_profit_q"].shift(4)
    stmt["total_assets_yoy"] = grouped["total_assets_q"].shift(4)
    stmt["equity_yoy"] = grouped["equity_q"].shift(4)
    stmt["revenue_yoy"] = grouped["revenue_q"].shift(4)
    stmt["report_date_yoy"] = grouped["report_date"].shift(4)
    return stmt.sort_values([Col.SYMBOL, Col.DATE, "report_date"])


def _align_latest_statement(
    statement_records: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    symbols: pd.Index,
) -> pd.DataFrame:
    frames = []
    for symbol, symbol_stmt in statement_records.groupby(Col.SYMBOL):
        if symbol not in symbols:
            continue
        symbol_stmt = symbol_stmt.sort_values(Col.DATE)
        right = symbol_stmt.drop_duplicates(Col.DATE, keep="last")
        left = pd.DataFrame({Col.DATE: signal_dates})
        aligned = pd.merge_asof(left, right, on=Col.DATE, direction="backward")
        aligned[Col.SYMBOL] = symbol
        frames.append(aligned)
    if not frames:
        index = pd.MultiIndex.from_product([signal_dates, symbols], names=[Col.DATE, Col.SYMBOL])
        return pd.DataFrame(index=index)
    aligned = pd.concat(frames, ignore_index=True)
    return aligned.set_index([Col.DATE, Col.SYMBOL]).sort_index()


def _compute_base_profitability_subfactors(aligned_statement: pd.DataFrame) -> dict[str, pd.DataFrame]:
    roeq = _safe_divide(aligned_statement["net_income_common"], aligned_statement["equity_q"])
    roaq = _safe_divide(aligned_statement["net_income_total"], aligned_statement["avg_assets_q"])
    gpoaq = _safe_divide(aligned_statement["gross_profit_q"], aligned_statement["avg_assets_q"])
    gmarq = _safe_divide(aligned_statement["gross_profit_q"], aligned_statement["revenue_q"])
    return {
        "ROEQ": roeq.unstack(Col.SYMBOL),
        "ROAQ": roaq.unstack(Col.SYMBOL),
        "GPOAQ": gpoaq.unstack(Col.SYMBOL),
        "GMARQ": gmarq.unstack(Col.SYMBOL),
    }


def _compute_growth_quality_subfactors(aligned_statement: pd.DataFrame) -> dict[str, pd.DataFrame]:
    droeq = _safe_divide(
        aligned_statement["net_income_common"] - aligned_statement["net_income_common_yoy"],
        aligned_statement["equity_yoy"],
    )
    droaq = _safe_divide(
        aligned_statement["net_income_total"] - aligned_statement["net_income_total_yoy"],
        aligned_statement["total_assets_yoy"],
    )
    dgpoaq = _safe_divide(
        aligned_statement["gross_profit_q"] - aligned_statement["gross_profit_yoy"],
        aligned_statement["total_assets_yoy"],
    )
    dgmarq = _safe_divide(
        aligned_statement["gross_profit_q"] - aligned_statement["gross_profit_yoy"],
        aligned_statement["revenue_yoy"],
    )
    return {
        "dROEQ": droeq.unstack(Col.SYMBOL),
        "dROAQ": droaq.unstack(Col.SYMBOL),
        "dGPOAQ": dgpoaq.unstack(Col.SYMBOL),
        "dGMARQ": dgmarq.unstack(Col.SYMBOL),
    }


def _extract_market_cap_panel(
    data: dict[str, pd.DataFrame],
    signal_dates: pd.DatetimeIndex,
    symbols: pd.Index,
) -> pd.DataFrame:
    for table_name in ("fundamental", "market"):
        source = data.get(table_name)
        if source is None or source.empty:
            continue
        source = _normalize_panel_index(source)
        column = _first_existing_column(source, MARKET_CAP_COLUMNS)
        if column is None:
            continue
        panel = pd.to_numeric(source[column], errors="coerce").unstack(Col.SYMBOL).sort_index()
        panel.index = pd.DatetimeIndex(pd.to_datetime(panel.index)).tz_localize(None)
        panel = panel.reindex(panel.index.union(signal_dates)).sort_index().ffill()
        return panel.reindex(index=signal_dates, columns=symbols)
    raise ValueError("final_quality_growth requires market cap column: market_cap / mkt_cap / total_mv / circ_mv")


def _extract_industry_panel(
    data: dict[str, pd.DataFrame],
    signal_dates: pd.DatetimeIndex,
    symbols: pd.Index,
) -> pd.DataFrame | None:
    for table_name in ("industry", "fundamental", "market"):
        source = data.get(table_name)
        if source is None or source.empty:
            continue
        if table_name == "industry":
            static_source = source.copy()
            if Col.SYMBOL in static_source.columns:
                static_source[Col.SYMBOL] = static_source[Col.SYMBOL].astype(str)
                static_source = static_source.set_index(Col.SYMBOL)
            if Col.SYMBOL in static_source.index.names and Col.DATE not in static_source.index.names:
                column = _first_existing_column(static_source, INDUSTRY_COLUMNS)
                if column is None:
                    continue
                if isinstance(static_source.index, pd.MultiIndex):
                    static_source = (
                        static_source.reset_index()
                        .drop_duplicates(Col.SYMBOL, keep="last")
                        .set_index(Col.SYMBOL)
                    )
                static_source.index = static_source.index.astype(str)
                values = static_source[~static_source.index.duplicated(keep="last")][column].reindex(symbols)
                return pd.DataFrame(
                    [values.to_numpy()] * len(signal_dates),
                    index=signal_dates,
                    columns=symbols,
                    dtype=object,
                )
        source = _normalize_panel_index(source)
        column = _first_existing_column(source, INDUSTRY_COLUMNS)
        if column is None:
            continue
        panel = source[column].astype("object").unstack(Col.SYMBOL).sort_index()
        panel.index = pd.DatetimeIndex(pd.to_datetime(panel.index)).tz_localize(None)
        panel = panel.reindex(panel.index.union(signal_dates)).sort_index().ffill()
        return panel.reindex(index=signal_dates, columns=symbols)
    return None


def _ttm_dividend_from_events(
    dividend: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    symbols: pd.Index,
) -> pd.DataFrame:
    if dividend is None or dividend.empty:
        return pd.DataFrame(np.nan, index=signal_dates, columns=symbols)
    div = _normalize_panel_index(dividend).reset_index()
    column = _first_existing_column(div, DIVIDEND_COLUMNS)
    if column is None:
        return pd.DataFrame(np.nan, index=signal_dates, columns=symbols)
    div[Col.DATE] = pd.to_datetime(div[Col.DATE]).dt.tz_localize(None)
    div[Col.SYMBOL] = div[Col.SYMBOL].astype(str)
    div["dividend_amount"] = pd.to_numeric(div[column], errors="coerce").abs()
    panel = pd.DataFrame(0.0, index=signal_dates, columns=symbols)
    for symbol, symbol_div in div.groupby(Col.SYMBOL):
        if symbol not in panel.columns:
            continue
        symbol_div = symbol_div.dropna(subset=["dividend_amount"]).sort_values(Col.DATE)
        values = []
        for dt in signal_dates:
            start = dt - pd.Timedelta(days=365)
            mask = symbol_div[Col.DATE].gt(start) & symbol_div[Col.DATE].le(dt)
            values.append(symbol_div.loc[mask, "dividend_amount"].sum())
        panel[symbol] = values
    panel.index.name = Col.DATE
    panel.columns.name = Col.SYMBOL
    return panel


def _ttm_dividend_from_statement(
    statement_records: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    symbols: pd.Index,
) -> pd.DataFrame:
    column = _first_existing_column(statement_records, ("dividends_paid", "pay_div"))
    if column is None:
        return pd.DataFrame(np.nan, index=signal_dates, columns=symbols)
    stmt = statement_records[[Col.DATE, Col.SYMBOL, column]].copy()
    stmt[column] = pd.to_numeric(stmt[column], errors="coerce").abs()
    panel = pd.DataFrame(np.nan, index=signal_dates, columns=symbols)
    for symbol, symbol_stmt in stmt.groupby(Col.SYMBOL):
        if symbol not in panel.columns:
            continue
        symbol_stmt = symbol_stmt.dropna(subset=[column]).sort_values(Col.DATE)
        values = []
        for dt in signal_dates:
            start = dt - pd.Timedelta(days=365)
            mask = symbol_stmt[Col.DATE].gt(start) & symbol_stmt[Col.DATE].le(dt)
            values.append(symbol_stmt.loc[mask, column].sum())
        panel[symbol] = values
    panel.index.name = Col.DATE
    panel.columns.name = Col.SYMBOL
    return panel


def _compute_dividend_ratio_ttm(
    data: dict[str, pd.DataFrame],
    statement_records: pd.DataFrame,
    market_cap: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    symbols: pd.Index,
) -> pd.DataFrame:
    dividend_amount = _ttm_dividend_from_events(data.get("dividend"), signal_dates, symbols)
    if dividend_amount.dropna(how="all").empty or dividend_amount.sum(axis=1, min_count=1).isna().all():
        dividend_amount = _ttm_dividend_from_statement(statement_records, signal_dates, symbols)
    result = dividend_amount.divide(market_cap.where(market_cap > 0))
    result.index.name = Col.DATE
    result.columns.name = Col.SYMBOL
    return result


def _neutralize_cross_section(values: pd.Series, log_market_cap: pd.Series, industry: pd.Series | None) -> pd.Series:
    valid = values.notna()
    if valid.sum() < 3:
        return values * np.nan
    y = values.loc[valid].astype(float)
    columns = [pd.Series(1.0, index=y.index, name="const")]
    size = log_market_cap.reindex(y.index)
    if size.notna().sum() >= 3 and size.nunique(dropna=True) > 1:
        columns.append(size.fillna(size.median()).rename("log_market_cap"))
    if industry is not None:
        industry_values = industry.reindex(y.index).fillna("UNKNOWN").astype(str)
        dummies = pd.get_dummies(industry_values, prefix="industry", drop_first=True, dtype=float)
        if not dummies.empty:
            columns.append(dummies)
    x = pd.concat(columns, axis=1)
    if x.shape[1] >= len(y):
        return y - y.mean()
    try:
        beta, *_ = np.linalg.lstsq(x.to_numpy(dtype=float), y.to_numpy(dtype=float), rcond=None)
    except np.linalg.LinAlgError:
        return y - y.mean()
    residual = y - x.to_numpy(dtype=float) @ beta
    return pd.Series(residual, index=y.index)


def _preprocess_subfactors(
    subfactors: dict[str, pd.DataFrame],
    market_cap: pd.DataFrame,
    industry: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    processed: dict[str, pd.DataFrame] = {}
    log_market_cap = np.log(market_cap.where(market_cap > 0))
    for name, factor in subfactors.items():
        factor = factor.reindex(index=market_cap.index, columns=market_cap.columns).replace([np.inf, -np.inf], np.nan)
        lower = factor.quantile(0.01, axis=1)
        upper = factor.quantile(0.99, axis=1)
        clipped = factor.clip(lower=lower, upper=upper, axis=0)
        filled = clipped.T.fillna(clipped.median(axis=1)).T
        mean = filled.mean(axis=1)
        std = filled.std(axis=1).replace(0, np.nan)
        zscore = filled.sub(mean, axis=0).div(std, axis=0)
        neutralized = pd.DataFrame(np.nan, index=zscore.index, columns=zscore.columns, dtype=float)
        for dt in zscore.index:
            industry_row = industry.loc[dt] if industry is not None and dt in industry.index else None
            residual = _neutralize_cross_section(zscore.loc[dt], log_market_cap.loc[dt], industry_row)
            neutralized.loc[dt, residual.index] = residual
        residual_mean = neutralized.mean(axis=1)
        residual_std = neutralized.std(axis=1).replace(0, np.nan)
        processed[name] = neutralized.sub(residual_mean, axis=0).div(residual_std, axis=0)
        processed[name].index.name = Col.DATE
        processed[name].columns.name = Col.SYMBOL
    return processed


def _valid_factor_dates(
    factors: dict[str, pd.DataFrame],
    min_valid_count: int,
) -> pd.DatetimeIndex:
    if not factors:
        return pd.DatetimeIndex([])
    dates = next(iter(factors.values())).index
    valid_dates = []
    for dt in dates:
        ready = True
        for factor in factors.values():
            row = factor.loc[dt].replace([np.inf, -np.inf], np.nan)
            if int(row.notna().sum()) < min_valid_count:
                ready = False
                break
        if ready:
            valid_dates.append(dt)
    return pd.DatetimeIndex(valid_dates)


def _reindex_factor_dict(
    factors: dict[str, pd.DataFrame],
    index: pd.DatetimeIndex,
    columns: pd.Index,
) -> dict[str, pd.DataFrame]:
    return {
        name: factor.reindex(index=index, columns=columns)
        for name, factor in factors.items()
    }


def _compute_forward_returns(market_data: pd.DataFrame, signal_dates: pd.DatetimeIndex, symbols: pd.Index) -> pd.DataFrame:
    close = market_data[Col.CLOSE].unstack(Col.SYMBOL).sort_index()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index)).tz_localize(None)
    close = close.reindex(close.index.union(signal_dates)).sort_index().ffill()
    close = close.reindex(index=signal_dates, columns=symbols)
    returns = close.pct_change(fill_method=None).shift(-1)
    returns.index.name = Col.DATE
    returns.columns.name = Col.SYMBOL
    return returns


def _rolling_rank_icir_weights(
    factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    names = list(factors)
    ic = pd.DataFrame(np.nan, index=forward_returns.index, columns=names, dtype=float)
    for name, factor in factors.items():
        aligned = factor.reindex_like(forward_returns)
        for dt in forward_returns.index:
            rows = pd.DataFrame({"factor": aligned.loc[dt], "ret": forward_returns.loc[dt]}).dropna()
            if len(rows) >= 5 and rows["factor"].nunique() > 1 and rows["ret"].nunique() > 1:
                ic.loc[dt, name] = rows["factor"].rank().corr(rows["ret"].rank())

    weights = pd.DataFrame(np.nan, index=forward_returns.index, columns=names, dtype=float)
    for pos, dt in enumerate(forward_returns.index):
        hist = ic.iloc[max(0, pos - window):pos]
        available_count = hist.count()
        mean = hist.mean(skipna=True)
        std = hist.std(skipna=True).replace(0, np.nan)
        icir = mean.divide(std).where(available_count >= min_periods)
        scores = icir.clip(lower=0).replace([np.inf, -np.inf], np.nan).dropna()
        if scores.sum() > 0:
            weights.loc[dt, scores.index] = scores / scores.sum()
        else:
            available = [
                name for name, factor in factors.items()
                if factor.loc[dt].notna().any()
            ]
            if available:
                weights.loc[dt, available] = 1.0 / len(available)
    weights.index.name = Col.DATE
    return weights


def _combine_weighted_factors(
    factors: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
    min_factors: int,
) -> pd.DataFrame:
    columns = next(iter(factors.values())).columns
    combined = pd.DataFrame(np.nan, index=weights.index, columns=columns, dtype=float)
    for dt in weights.index:
        numerator = pd.Series(0.0, index=columns, dtype=float)
        denominator = pd.Series(0.0, index=columns, dtype=float)
        counts = pd.Series(0, index=columns, dtype=int)
        for name, weight in weights.loc[dt].dropna().items():
            if weight <= 0 or name not in factors:
                continue
            row = factors[name].loc[dt].reindex(columns)
            mask = row.notna()
            numerator.loc[mask] += row.loc[mask] * weight
            denominator.loc[mask] += weight
            counts.loc[mask] += 1
        valid = denominator.gt(0) & counts.ge(min_factors)
        valid_columns = valid.index[valid]
        combined.loc[dt, valid_columns] = (
            numerator.loc[valid_columns] / denominator.loc[valid_columns]
        ).to_numpy()
    combined.index.name = Col.DATE
    combined.columns.name = Col.SYMBOL
    return combined


def _stabilize_weight_matrix(
    weights: pd.DataFrame,
    factors: dict[str, pd.DataFrame],
    min_positive_weights: int,
) -> pd.DataFrame:
    stabilized = weights.copy()
    for dt in stabilized.index:
        positive = stabilized.loc[dt].dropna()
        positive = positive[positive > 0]
        if len(positive) >= min_positive_weights:
            continue
        available = [
            name for name, factor in factors.items()
            if dt in factor.index and factor.loc[dt].notna().any()
        ]
        stabilized.loc[dt] = np.nan
        if len(available) >= min_positive_weights:
            stabilized.loc[dt, available] = 1.0 / len(available)
    return stabilized


def _compute_quality(
    processed_subfactors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    window: int,
    min_periods: int,
    min_factors: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = _rolling_rank_icir_weights(processed_subfactors, forward_returns, window, min_periods)
    weights = _stabilize_weight_matrix(weights, processed_subfactors, min_factors)
    quality = _combine_weighted_factors(processed_subfactors, weights, min_factors)
    return quality, weights


def _compute_equal_weight_quality(
    processed_subfactors: dict[str, pd.DataFrame],
    min_factors: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = next(iter(processed_subfactors.values())).index
    names = list(processed_subfactors)
    weights = pd.DataFrame(np.nan, index=dates, columns=names, dtype=float)
    for dt in dates:
        available = [
            name for name, factor in processed_subfactors.items()
            if factor.loc[dt].notna().any()
        ]
        if len(available) >= min_factors:
            weights.loc[dt, available] = 1.0 / len(available)
    weights.index.name = Col.DATE
    quality = _combine_weighted_factors(processed_subfactors, weights, min_factors)
    return quality, weights


def _compute_quality_increase(
    quality: pd.DataFrame,
    aligned_statement: pd.DataFrame,
    lag_mode: str,
) -> pd.DataFrame:
    if lag_mode == "4q":
        result = quality - quality.shift(4)
    elif lag_mode == "12m":
        result = quality - quality.shift(12)
    else:
        result = pd.DataFrame(np.nan, index=quality.index, columns=quality.columns, dtype=float)
        report_panel = aligned_statement["report_date"].unstack(Col.SYMBOL).reindex_like(quality)
        yoy_report_panel = aligned_statement["report_date_yoy"].unstack(Col.SYMBOL).reindex_like(quality)
        long_quality = quality.stack(future_stack=True).rename("quality").reset_index()
        long_report = report_panel.stack(future_stack=True).rename("report_date").reset_index()
        quality_by_report = long_quality.merge(long_report, on=[Col.DATE, Col.SYMBOL], how="left")
        quality_by_report = quality_by_report.dropna(subset=["quality", "report_date"])
        quality_by_report = quality_by_report.sort_values([Col.SYMBOL, "report_date", Col.DATE])
        quality_by_report = quality_by_report.drop_duplicates([Col.SYMBOL, "report_date"], keep="last")
        lookup = quality_by_report.set_index([Col.SYMBOL, "report_date"])["quality"]
        for dt in quality.index:
            for symbol in quality.columns:
                current = quality.loc[dt, symbol]
                prior_report = yoy_report_panel.loc[dt, symbol]
                if pd.isna(current) or pd.isna(prior_report):
                    continue
                key = (symbol, pd.Timestamp(prior_report))
                if key in lookup.index:
                    result.loc[dt, symbol] = current - lookup.loc[key]
        if result.dropna(how="all").empty:
            result = quality - quality.shift(4)
    result.index.name = Col.DATE
    result.columns.name = Col.SYMBOL
    return result.replace([np.inf, -np.inf], np.nan)


def _compute_final_quality_growth(
    quality: pd.DataFrame,
    quality_increase: pd.DataFrame,
    forward_returns: pd.DataFrame,
    window: int,
    min_periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    quality_z = _preprocess_subfactors(
        {"Quality": quality, "QualityIncrease": quality_increase},
        pd.DataFrame(1.0, index=quality.index, columns=quality.columns),
        None,
    )
    weights = _rolling_rank_icir_weights(quality_z, forward_returns, window, min_periods)
    final_factor = _combine_weighted_factors(quality_z, weights, min_factors=1)
    return final_factor, weights


def _empty_factor(signal_dates: pd.DatetimeIndex, symbols: pd.Index) -> pd.DataFrame:
    factor = pd.DataFrame(np.nan, index=signal_dates, columns=symbols, dtype=float)
    factor.index.name = Col.DATE
    factor.columns.name = Col.SYMBOL
    return factor


def _prepare_quality_inputs(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, pd.Index, pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    market_data = _normalize_panel_index(data["market"])
    statement_records = _prepare_statement_records(data.get("statement"))
    signal_dates = _resolve_signal_dates(data, market_data)
    symbols = market_data.index.get_level_values(Col.SYMBOL).unique()
    market_cap = _extract_market_cap_panel(data, signal_dates, symbols)
    industry = _extract_industry_panel(data, signal_dates, symbols)
    aligned_statement = _align_latest_statement(statement_records, signal_dates, symbols)
    return market_data, statement_records, signal_dates, symbols, market_cap, industry, aligned_statement


def _build_selected_subfactors(
    selected_names: tuple[str, ...],
    data: dict[str, pd.DataFrame],
    statement_records: pd.DataFrame,
    aligned_statement: pd.DataFrame,
    market_cap: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    symbols: pd.Index,
) -> dict[str, pd.DataFrame]:
    subfactors = {}
    if any(name in selected_names for name in BASE_PROFITABILITY_NAMES):
        subfactors.update(_compute_base_profitability_subfactors(aligned_statement))
    if any(name in selected_names for name in GROWTH_QUALITY_NAMES):
        subfactors.update(_compute_growth_quality_subfactors(aligned_statement))
    if "DividendRatioTTM" in selected_names:
        subfactors["DividendRatioTTM"] = _compute_dividend_ratio_ttm(
            data,
            statement_records,
            market_cap,
            signal_dates,
            symbols,
        )
    return _reindex_factor_dict(
        {name: subfactors[name] for name in selected_names},
        signal_dates,
        symbols,
    )


def _generate_quality_growth_signal(
    factor: "FinalQualityGrowth",
    data: dict[str, pd.DataFrame],
    equal_subfactor_weights: bool,
) -> pd.DataFrame:
    (
        market_data,
        statement_records,
        signal_dates,
        symbols,
        market_cap,
        industry,
        aligned_statement,
    ) = _prepare_quality_inputs(data)
    subfactors = _build_selected_subfactors(
        SUBFACTOR_NAMES,
        data,
        statement_records,
        aligned_statement,
        market_cap,
        signal_dates,
        symbols,
    )
    final_factor = _empty_factor(signal_dates, symbols)

    subfactor_ready_dates = _valid_factor_dates(subfactors, factor.min_valid_cross_section)
    if subfactor_ready_dates.empty:
        factor.last_statement = aligned_statement
        factor.last_subfactors = subfactors
        factor.last_processed_subfactors = {}
        factor.last_quality = final_factor.copy()
        factor.last_quality_weights = pd.DataFrame(index=signal_dates, columns=SUBFACTOR_NAMES, dtype=float)
        factor.last_quality_increase = final_factor.copy()
        factor.last_final_weights = pd.DataFrame(index=signal_dates, columns=["Quality", "QualityIncrease"], dtype=float)
        return final_factor

    full_forward_returns = _compute_forward_returns(market_data, signal_dates, symbols)
    ready_subfactors = _reindex_factor_dict(subfactors, subfactor_ready_dates, symbols)
    ready_market_cap = market_cap.reindex(index=subfactor_ready_dates, columns=symbols)
    ready_industry = industry.reindex(index=subfactor_ready_dates, columns=symbols) if industry is not None else None
    processed = _preprocess_subfactors(ready_subfactors, ready_market_cap, ready_industry)
    forward_returns = full_forward_returns.reindex(index=subfactor_ready_dates, columns=symbols)
    if equal_subfactor_weights:
        quality, quality_weights = _compute_equal_weight_quality(processed, factor.quality_min_factors)
    else:
        quality, quality_weights = _compute_quality(
            processed,
            forward_returns,
            factor.rank_icir_window,
            factor.rank_icir_min_periods,
            factor.quality_min_factors,
        )
    quality_increase = _compute_quality_increase(
        quality,
        aligned_statement,
        factor.quality_increase_lag,
    )
    final_inputs = {"Quality": quality, "QualityIncrease": quality_increase}
    final_ready_dates = _valid_factor_dates(final_inputs, factor.min_valid_cross_section)
    if final_ready_dates.empty:
        final_weights = pd.DataFrame(index=subfactor_ready_dates, columns=["Quality", "QualityIncrease"], dtype=float)
    else:
        ready_quality = quality.reindex(index=final_ready_dates, columns=symbols)
        ready_quality_increase = quality_increase.reindex(index=final_ready_dates, columns=symbols)
        ready_forward_returns = forward_returns.reindex(index=final_ready_dates, columns=symbols)
        scored_factor, final_weights = _compute_final_quality_growth(
            ready_quality,
            ready_quality_increase,
            ready_forward_returns,
            factor.rank_icir_window,
            factor.rank_icir_min_periods,
        )
        final_factor.loc[scored_factor.index, scored_factor.columns] = scored_factor

    factor.last_statement = aligned_statement
    factor.last_subfactors = subfactors
    factor.last_processed_subfactors = processed
    factor.last_quality = quality
    factor.last_quality_weights = quality_weights
    factor.last_quality_increase = quality_increase
    factor.last_final_weights = final_weights
    final_factor.index.name = Col.DATE
    final_factor.columns.name = Col.SYMBOL
    return final_factor


def _generate_rank_ic_subfactor_signal(
    factor: "FinalQualityGrowth",
    data: dict[str, pd.DataFrame],
    selected_names: tuple[str, ...],
) -> pd.DataFrame:
    (
        market_data,
        statement_records,
        signal_dates,
        symbols,
        market_cap,
        industry,
        aligned_statement,
    ) = _prepare_quality_inputs(data)
    subfactors = _build_selected_subfactors(
        selected_names,
        data,
        statement_records,
        aligned_statement,
        market_cap,
        signal_dates,
        symbols,
    )
    final_factor = _empty_factor(signal_dates, symbols)
    subfactor_ready_dates = _valid_factor_dates(subfactors, factor.min_valid_cross_section)
    if subfactor_ready_dates.empty:
        factor.last_statement = aligned_statement
        factor.last_subfactors = subfactors
        factor.last_processed_subfactors = {}
        factor.last_quality = final_factor.copy()
        factor.last_quality_weights = pd.DataFrame(index=signal_dates, columns=selected_names, dtype=float)
        return final_factor

    full_forward_returns = _compute_forward_returns(market_data, signal_dates, symbols)
    ready_subfactors = _reindex_factor_dict(subfactors, subfactor_ready_dates, symbols)
    ready_market_cap = market_cap.reindex(index=subfactor_ready_dates, columns=symbols)
    ready_industry = industry.reindex(index=subfactor_ready_dates, columns=symbols) if industry is not None else None
    processed = _preprocess_subfactors(ready_subfactors, ready_market_cap, ready_industry)
    forward_returns = full_forward_returns.reindex(index=subfactor_ready_dates, columns=symbols)
    quality, quality_weights = _compute_quality(
        processed,
        forward_returns,
        factor.rank_icir_window,
        factor.rank_icir_min_periods,
        factor.quality_min_factors,
    )
    final_factor.loc[quality.index, quality.columns] = quality
    factor.last_statement = aligned_statement
    factor.last_subfactors = subfactors
    factor.last_processed_subfactors = processed
    factor.last_quality = quality
    factor.last_quality_weights = quality_weights
    return final_factor


@register_factor
class FinalQualityGrowth(BaseFactor):
    name = "final_quality_growth"
    description = "Quality and quality-increase composite weighted by rolling RankICIR"
    category = "quality"

    rank_icir_window = 24
    rank_icir_min_periods = 6
    quality_min_factors = 3
    quality_increase_lag = "yoy_report"
    min_valid_cross_section = 5

    def __init__(self, quality_increase_lag: str | None = None) -> None:
        if quality_increase_lag is not None:
            if quality_increase_lag not in {"yoy_report", "4q", "12m"}:
                raise ValueError("quality_increase_lag must be one of: yoy_report, 4q, 12m")
            self.quality_increase_lag = quality_increase_lag

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        return _generate_quality_growth_signal(self, data, equal_subfactor_weights=False)


@register_factor
class FinalQualityGrowthEqualWeight(FinalQualityGrowth):
    name = "final_quality_growth_equal_weight"
    description = "Quality growth composite with equal-weighted quality subfactors"
    category = "quality"

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        return _generate_quality_growth_signal(self, data, equal_subfactor_weights=True)


@register_factor
class BaseProfitabilityRankIC(FinalQualityGrowth):
    name = "base_profitability_rank_ic"
    description = "Base profitability quality subfactors weighted by rolling RankICIR"
    category = "quality"

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        return _generate_rank_ic_subfactor_signal(self, data, BASE_PROFITABILITY_NAMES)


@register_factor
class GrowthQualityRankIC(FinalQualityGrowth):
    name = "growth_quality_rank_ic"
    description = "Growth quality subfactors weighted by rolling RankICIR"
    category = "quality"

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        return _generate_rank_ic_subfactor_signal(self, data, GROWTH_QUALITY_NAMES)
