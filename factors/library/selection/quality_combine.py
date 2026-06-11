"""Quality + Quality-Growth composite stock-selection factor.

Implements a multi-factor module combining base profitability, growth quality,
and dividend return subfactors into a single composite signal.  All financial
metrics use *single-quarter* (Q) figures to maximise timeliness and a strict
point-in-time discipline: financial statements only become visible after their
publication / filing date, aligned to monthly rebalance dates via
``pd.merge_asof(..., direction='backward')``.

Pipeline
--------
1. **Base profitability subfactors** — ROEQ, ROAQ, GPOAQ, GMARQ
2. **Growth quality subfactors** — dROEQ, dROAQ, dGPOAQ, dGMARQ
   (change-over-level to avoid extreme YoY% when base is near zero)
3. **Dividend return** — DividendRatioTTM
4. **Cross-section preprocessing** — winsorise → z-score → neutralise on
   industry dummies + log(market cap)
5. **Quality** — rolling RankICIR weighted composite of 9 subfactors
6. **QualityIncrease** — Quality_t − Quality_{t-1} (YoY comparable)
7. **FinalQualityGrowth** — rolling RankICIR weighted blend of
   zscore(Quality) and zscore(QualityIncrease)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from data.schema import Col
from factors.base import BaseFactor
from factors.registry import register_factor

# ── column candidates for flexible data sources ────────────────────────── #

SUBFACTOR_NAMES = (
    "ROEQ", "ROAQ", "GPOAQ", "GMARQ",
    "dROEQ", "dROAQ", "dGPOAQ", "dGMARQ",
    "DividendRatioTTM",
)
BASE_PROFITABILITY = ("ROEQ", "ROAQ", "GPOAQ", "GMARQ")
GROWTH_QUALITY = ("dROEQ", "dROAQ", "dGPOAQ", "dGMARQ")

MKT_CAP_COLS = ("market_cap", Col.MKT_CAP, "total_mv", "circ_mv")
INDUSTRY_COLS = (
    "industry", "sector", "gics_sector", "gics_industry",
    "sw_industry", "industry_code",
)
DIVIDEND_COLS = (
    "dividend", "dividend_amount", "cash_dividend",
    "amount", "dividends_paid", "pay_div",
)


# ── small helpers ──────────────────────────────────────────────────────── #

def _first_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first column name from *candidates* that exists in *df*."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _num_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    """Return the first matching column coerced to numeric, or all-NaN."""
    c = _first_col(df, candidates)
    if c is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[c], errors="coerce")


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Division that returns NaN when denominator ≈ 0."""
    return num.divide(den.where(den.abs() > 1e-12))


def _norm_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure MultiIndex(date, symbol) with proper dtypes."""
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Expected MultiIndex[date, symbol] data")
    names = list(df.index.names)
    d_lvl = names.index(Col.DATE) if Col.DATE in names else 0
    s_lvl = names.index(Col.SYMBOL) if Col.SYMBOL in names else 1
    dates = pd.DatetimeIndex(pd.to_datetime(df.index.get_level_values(d_lvl))).tz_localize(None)
    syms = df.index.get_level_values(s_lvl).astype(str)
    out = df.copy()
    out.index = pd.MultiIndex.from_arrays([dates, syms], names=[Col.DATE, Col.SYMBOL])
    return out.sort_index()


# ── signal date resolution ─────────────────────────────────────────────── #

def _signal_dates(data: dict[str, pd.DataFrame], mkt: pd.DataFrame) -> pd.DatetimeIndex:
    sd = data.get("signal_dates")
    if sd is None:
        dates = mkt.index.get_level_values(Col.DATE).unique()
    elif isinstance(sd, pd.DataFrame):
        dates = sd[Col.DATE] if Col.DATE in sd.columns else sd.index
    elif isinstance(sd, (pd.Series, pd.DatetimeIndex, pd.Index)):
        dates = sd
    else:
        dates = sd
    return pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(None).unique().sort_values()


# ── Step 0: Prepare statement records ─────────────────────────────────── #

def _prepare_statement(statement: pd.DataFrame) -> pd.DataFrame:
    """Normalise and enrich the raw *statement* table.

    Adds YoY lag columns (shift-4) for growth quality computation.
    Single-quarter metrics are preserved directly; average total assets
    are computed as (本期 + 上期) / 2.
    """
    if statement is None or statement.empty:
        raise ValueError("quality_combine requires data['statement']")

    stmt = _norm_panel(statement).reset_index()
    stmt[Col.DATE] = pd.to_datetime(stmt[Col.DATE]).dt.tz_localize(None)

    # report_date
    if "report_date" in stmt.columns:
        stmt["report_date"] = pd.to_datetime(stmt["report_date"], errors="coerce").dt.tz_localize(None)
    else:
        stmt["report_date"] = stmt[Col.DATE]
    stmt["report_date"] = stmt["report_date"].fillna(stmt[Col.DATE])
    stmt["statement_date"] = stmt[Col.DATE]
    stmt[Col.SYMBOL] = stmt[Col.SYMBOL].astype(str)

    # Filter out annual-only rows (fiscal_quarter == 0)
    if "fiscal_quarter" in stmt.columns:
        qtr = pd.to_numeric(stmt["fiscal_quarter"], errors="coerce")
        stmt = stmt.loc[qtr.ne(0) | qtr.isna()].copy()

    # Standardised column aliases
    stmt["net_income_common"] = _num_col(stmt, ("net_income_common_stock", "net_profit", "net_income"))
    stmt["net_income_total"] = _num_col(stmt, ("net_income", "net_profit"))
    stmt["revenue_q"] = _num_col(stmt, ("revenue", "total_revenue"))
    stmt["cost_q"] = _num_col(stmt, ("cost_revenue", "operating_cost", "cost"))
    stmt["gross_profit_q"] = _num_col(stmt, ("gross_profit",))
    stmt["gross_profit_q"] = stmt["gross_profit_q"].where(
        stmt["gross_profit_q"].notna(),
        stmt["revenue_q"] - stmt["cost_q"],
    )
    stmt["total_assets_q"] = _num_col(stmt, ("total_assets",))
    stmt["equity_q"] = _num_col(stmt, ("shareholder_equity", "equity"))

    stmt = stmt.dropna(subset=[Col.DATE, Col.SYMBOL])
    stmt = stmt.sort_values([Col.SYMBOL, Col.DATE, "report_date"])
    stmt = stmt.drop_duplicates([Col.SYMBOL, Col.DATE], keep="last")

    grp = stmt.groupby(Col.SYMBOL, group_keys=False)
    stmt["avg_assets_q"] = (stmt["total_assets_q"] + grp["total_assets_q"].shift(1)) / 2.0
    # YoY comparable (4 quarters ago)
    for col in (
        "net_income_common", "net_income_total", "gross_profit_q",
        "total_assets_q", "equity_q", "revenue_q", "report_date",
    ):
        stmt[f"{col}_yoy"] = grp[col].shift(4)

    return stmt.sort_values([Col.SYMBOL, Col.DATE, "report_date"])


def _align_latest(
    stmt: pd.DataFrame,
    sig_dates: pd.DatetimeIndex,
    symbols: pd.Index,
) -> pd.DataFrame:
    """Point-in-time alignment: for each signal date, take the latest
    available statement (backward ``merge_asof``) so that only already-published
    financial data is used — strict no-look-ahead."""
    frames = []
    for sym, sym_stmt in stmt.groupby(Col.SYMBOL):
        if sym not in symbols:
            continue
        right = sym_stmt.sort_values(Col.DATE).drop_duplicates(Col.DATE, keep="last")
        left = pd.DataFrame({Col.DATE: sig_dates})
        aligned = pd.merge_asof(left, right, on=Col.DATE, direction="backward")
        aligned[Col.SYMBOL] = sym
        frames.append(aligned)
    if not frames:
        idx = pd.MultiIndex.from_product([sig_dates, symbols], names=[Col.DATE, Col.SYMBOL])
        return pd.DataFrame(index=idx)
    out = pd.concat(frames, ignore_index=True)
    return out.set_index([Col.DATE, Col.SYMBOL]).sort_index()


# ── Step 1: Base profitability subfactors ──────────────────────────────── #

def _base_profitability(aligned: pd.DataFrame) -> dict[str, pd.DataFrame]:
    roeq = _safe_div(aligned["net_income_common"], aligned["equity_q"])
    roaq = _safe_div(aligned["net_income_total"], aligned["avg_assets_q"])
    gpoaq = _safe_div(aligned["gross_profit_q"], aligned["avg_assets_q"])
    gmarq = _safe_div(aligned["gross_profit_q"], aligned["revenue_q"])
    return {
        "ROEQ": roeq.unstack(Col.SYMBOL),
        "ROAQ": roaq.unstack(Col.SYMBOL),
        "GPOAQ": gpoaq.unstack(Col.SYMBOL),
        "GMARQ": gmarq.unstack(Col.SYMBOL),
    }


# ── Step 2: Growth quality subfactors ──────────────────────────────────── #
# Change-over-level form: Δ = (current − YoY) / YoY_base
# Avoids division by near-zero base that plagues simple YoY% ratios.

def _growth_quality(aligned: pd.DataFrame) -> dict[str, pd.DataFrame]:
    droeq = _safe_div(
        aligned["net_income_common"] - aligned["net_income_common_yoy"],
        aligned["equity_q_yoy"],
    )
    droaq = _safe_div(
        aligned["net_income_total"] - aligned["net_income_total_yoy"],
        aligned["total_assets_q_yoy"],
    )
    dgpoaq = _safe_div(
        aligned["gross_profit_q"] - aligned["gross_profit_q_yoy"],
        aligned["total_assets_q_yoy"],
    )
    dgmarq = _safe_div(
        aligned["gross_profit_q"] - aligned["gross_profit_q_yoy"],
        aligned["revenue_q_yoy"],
    )
    return {
        "dROEQ": droeq.unstack(Col.SYMBOL),
        "dROAQ": droaq.unstack(Col.SYMBOL),
        "dGPOAQ": dgpoaq.unstack(Col.SYMBOL),
        "dGMARQ": dgmarq.unstack(Col.SYMBOL),
    }


# ── Step 3: Dividend ratio TTM ────────────────────────────────────────── #

def _ttm_dividend_events(
    div: pd.DataFrame | None,
    sig_dates: pd.DatetimeIndex,
    symbols: pd.Index,
) -> pd.DataFrame:
    """Sum of cash dividends over the trailing 365 days from dividend events."""
    if div is None or div.empty:
        return pd.DataFrame(np.nan, index=sig_dates, columns=symbols)
    d = _norm_panel(div).reset_index()
    col = _first_col(d, DIVIDEND_COLS)
    if col is None:
        return pd.DataFrame(np.nan, index=sig_dates, columns=symbols)
    d[Col.DATE] = pd.to_datetime(d[Col.DATE]).dt.tz_localize(None)
    d[Col.SYMBOL] = d[Col.SYMBOL].astype(str)
    d["amt"] = pd.to_numeric(d[col], errors="coerce").abs()
    panel = pd.DataFrame(0.0, index=sig_dates, columns=symbols)
    for sym, sg in d.groupby(Col.SYMBOL):
        if sym not in panel.columns:
            continue
        sg = sg.dropna(subset=["amt"]).sort_values(Col.DATE)
        vals = []
        for dt in sig_dates:
            mask = sg[Col.DATE].gt(dt - pd.Timedelta(days=365)) & sg[Col.DATE].le(dt)
            vals.append(sg.loc[mask, "amt"].sum())
        panel[sym] = vals
    panel.index.name = Col.DATE
    panel.columns.name = Col.SYMBOL
    return panel


def _ttm_dividend_stmt(
    stmt: pd.DataFrame,
    sig_dates: pd.DatetimeIndex,
    symbols: pd.Index,
) -> pd.DataFrame:
    """Sum of dividends-paid over the trailing 365 days from statement records."""
    col = _first_col(stmt, ("dividends_paid", "pay_div"))
    if col is None:
        return pd.DataFrame(np.nan, index=sig_dates, columns=symbols)
    s = stmt[[Col.DATE, Col.SYMBOL, col]].copy()
    s[col] = pd.to_numeric(s[col], errors="coerce").abs()
    panel = pd.DataFrame(np.nan, index=sig_dates, columns=symbols)
    for sym, sg in s.groupby(Col.SYMBOL):
        if sym not in panel.columns:
            continue
        sg = sg.dropna(subset=[col]).sort_values(Col.DATE)
        vals = []
        for dt in sig_dates:
            mask = sg[Col.DATE].gt(dt - pd.Timedelta(days=365)) & sg[Col.DATE].le(dt)
            vals.append(sg.loc[mask, col].sum())
        panel[sym] = vals
    panel.index.name = Col.DATE
    panel.columns.name = Col.SYMBOL
    return panel


def _dividend_ratio_ttm(
    data: dict[str, pd.DataFrame],
    stmt: pd.DataFrame,
    mkt_cap: pd.DataFrame,
    sig_dates: pd.DatetimeIndex,
    symbols: pd.Index,
) -> pd.DataFrame:
    """DividendRatioTTM = trailing-12-month dividends / current market cap."""
    div_amt = _ttm_dividend_events(data.get("dividend"), sig_dates, symbols)
    if div_amt.dropna(how="all").empty or div_amt.sum(axis=1, min_count=1).isna().all():
        div_amt = _ttm_dividend_stmt(stmt, sig_dates, symbols)
    result = div_amt.divide(mkt_cap.where(mkt_cap > 0))
    result.index.name = Col.DATE
    result.columns.name = Col.SYMBOL
    return result


# ── Market cap & industry panels ──────────────────────────────────────── #

def _market_cap_panel(
    data: dict[str, pd.DataFrame],
    sig_dates: pd.DatetimeIndex,
    symbols: pd.Index,
) -> pd.DataFrame:
    for tbl in ("fundamental", "market"):
        src = data.get(tbl)
        if src is None or src.empty:
            continue
        src = _norm_panel(src)
        col = _first_col(src, MKT_CAP_COLS)
        if col is None:
            continue
        panel = pd.to_numeric(src[col], errors="coerce").unstack(Col.SYMBOL).sort_index()
        panel.index = pd.DatetimeIndex(pd.to_datetime(panel.index)).tz_localize(None)
        panel = panel.reindex(panel.index.union(sig_dates)).sort_index().ffill()
        return panel.reindex(index=sig_dates, columns=symbols)
    raise ValueError(
        "quality_combine requires a market-cap column: "
        "market_cap / mkt_cap / total_mv / circ_mv"
    )


def _industry_panel(
    data: dict[str, pd.DataFrame],
    sig_dates: pd.DatetimeIndex,
    symbols: pd.Index,
) -> pd.DataFrame | None:
    # Static industry table (indexed by symbol, no date)
    static = data.get("industry")
    if static is not None and not static.empty:
        col = _first_col(static, INDUSTRY_COLS)
        if col is not None:
            if Col.SYMBOL in static.columns:
                static = static.set_index(Col.SYMBOL)
            static.index = static.index.astype(str)
            vals = (
                static[~static.index.duplicated(keep="last")][col]
                .reindex(symbols)
            )
            return pd.DataFrame(
                [vals.to_numpy()] * len(sig_dates),
                index=sig_dates,
                columns=symbols,
                dtype=object,
            )
    # Time-varying industry from fundamental / market panels
    for tbl in ("fundamental", "market"):
        src = data.get(tbl)
        if src is None or src.empty:
            continue
        src = _norm_panel(src)
        col = _first_col(src, INDUSTRY_COLS)
        if col is None:
            continue
        panel = src[col].astype("object").unstack(Col.SYMBOL).sort_index()
        panel.index = pd.DatetimeIndex(pd.to_datetime(panel.index)).tz_localize(None)
        panel = panel.reindex(panel.index.union(sig_dates)).sort_index().ffill()
        return panel.reindex(index=sig_dates, columns=symbols)
    return None


# ── Step 4: Cross-section preprocessing ───────────────────────────────── #

def _neutralize_xs(
    values: pd.Series,
    log_mcap: pd.Series,
    industry: pd.Series | None,
) -> pd.Series:
    """OLS neutralise against log(market cap) + industry dummies, return residuals."""
    valid = values.notna()
    if valid.sum() < 3:
        return values * np.nan
    y = values.loc[valid].astype(float)
    parts = [pd.Series(1.0, index=y.index, name="const")]
    size = log_mcap.reindex(y.index)
    if size.notna().sum() >= 3 and size.nunique(dropna=True) > 1:
        parts.append(size.fillna(size.median()).rename("log_mcap"))
    # Split stocks into two groups:
    #   has_industry  → neutralise on log_mcap + industry dummies
    #   no_industry   → neutralise on log_mcap only (skip industry)
    if industry is not None:
        ind_raw = industry.reindex(y.index)
        has_ind = ind_raw.notna()
    else:
        has_ind = pd.Series(False, index=y.index)

    result = pd.Series(np.nan, index=y.index, dtype=float)

    for mask, include_industry in ((has_ind, True), (~has_ind & valid, False)):
        if mask.sum() < 3:
            if mask.sum() > 0:
                result.loc[mask] = y.loc[mask] - y.loc[mask].mean()
            continue
        y_sub = y.loc[mask]
        parts_sub = [pd.Series(1.0, index=y_sub.index, name="const")]
        size_sub = log_mcap.reindex(y_sub.index)
        if size_sub.notna().sum() >= 3 and size_sub.nunique(dropna=True) > 1:
            parts_sub.append(size_sub.fillna(size_sub.median()).rename("log_mcap"))
        if include_industry:
            ind_sub = ind_raw.reindex(y_sub.index).astype(str)
            dummies = pd.get_dummies(ind_sub, prefix="ind", drop_first=True, dtype=float)
            if not dummies.empty:
                parts_sub.append(dummies)
        X_sub = pd.concat(parts_sub, axis=1)
        if X_sub.shape[1] >= len(y_sub):
            result.loc[mask] = y_sub - y_sub.mean()
        else:
            try:
                beta, *_ = np.linalg.lstsq(X_sub.to_numpy(float), y_sub.to_numpy(float), rcond=None)
            except np.linalg.LinAlgError:
                result.loc[mask] = y_sub - y_sub.mean()
            else:
                result.loc[mask] = y_sub.to_numpy(float) - X_sub.to_numpy(float) @ beta

    return result


def _preprocess(
    subs: dict[str, pd.DataFrame],
    mcap: pd.DataFrame,
    ind: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    """Winsorise → z-score → industry/size neutralise → re-zscore."""
    log_mcap = np.log(mcap.where(mcap > 0))
    out: dict[str, pd.DataFrame] = {}
    for name, f in subs.items():
        f = f.reindex(index=mcap.index, columns=mcap.columns).replace([np.inf, -np.inf], np.nan)
        # Winsorise at 1%/99%
        lo = f.quantile(0.01, axis=1)
        hi = f.quantile(0.99, axis=1)
        f = f.clip(lower=lo, upper=hi, axis=0)
        # Fill NaN with cross-section median
        f = f.T.fillna(f.median(axis=1)).T
        # Z-score
        mu = f.mean(axis=1)
        sigma = f.std(axis=1).replace(0, np.nan)
        z = f.sub(mu, axis=0).div(sigma, axis=0)
        # Neutralise
        neut = pd.DataFrame(np.nan, index=z.index, columns=z.columns, dtype=float)
        for dt in z.index:
            ind_row = ind.loc[dt] if ind is not None and dt in ind.index else None
            res = _neutralize_xs(z.loc[dt], log_mcap.loc[dt], ind_row)
            neut.loc[dt, res.index] = res
        # Re-zscore residuals
        rmu = neut.mean(axis=1)
        rsig = neut.std(axis=1).replace(0, np.nan)
        out[name] = neut.sub(rmu, axis=0).div(rsig, axis=0)
        out[name].index.name = Col.DATE
        out[name].columns.name = Col.SYMBOL
    return out


# ── Forward returns & Rank IC helpers ──────────────────────────────────── #

def _forward_returns(
    mkt: pd.DataFrame,
    sig_dates: pd.DatetimeIndex,
    symbols: pd.Index,
) -> pd.DataFrame:
    close = mkt[Col.CLOSE].unstack(Col.SYMBOL).sort_index()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index)).tz_localize(None)
    close = close.reindex(close.index.union(sig_dates)).sort_index().ffill()
    close = close.reindex(index=sig_dates, columns=symbols)
    ret = close.pct_change(fill_method=None).shift(-1)
    ret.index.name = Col.DATE
    ret.columns.name = Col.SYMBOL
    return ret


def _valid_dates(
    factors: dict[str, pd.DataFrame],
    min_count: int,
) -> pd.DatetimeIndex:
    if not factors:
        return pd.DatetimeIndex([])
    dates = next(iter(factors.values())).index
    ok = []
    for dt in dates:
        ready = True
        for f in factors.values():
            if f.loc[dt].replace([np.inf, -np.inf], np.nan).notna().sum() < min_count:
                ready = False
                break
        if ready:
            ok.append(dt)
    return pd.DatetimeIndex(ok)


def _reindex_factors(
    factors: dict[str, pd.DataFrame],
    idx: pd.DatetimeIndex,
    cols: pd.Index,
) -> dict[str, pd.DataFrame]:
    return {n: f.reindex(index=idx, columns=cols) for n, f in factors.items()}


# ── Step 5: Quality (rolling RankICIR weighted) ────────────────────────── #

def _rolling_rank_icir_weights(
    factors: dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    """Compute rolling RankICIR for each subfactor, return normalised weights.

    RankIC = Spearman rank correlation between factor value and next-period
    return.  RankICIR = mean(RankIC) / std(RankIC) over a rolling window.
    Weight ∝ max(RankICIR, 0), normalised to sum to 1.  Falls back to equal
    weight when the history is insufficient.
    """
    names = list(factors)
    ic = pd.DataFrame(np.nan, index=fwd_ret.index, columns=names, dtype=float)
    for name, f in factors.items():
        aligned = f.reindex_like(fwd_ret)
        for dt in fwd_ret.index:
            rows = pd.DataFrame({"f": aligned.loc[dt], "r": fwd_ret.loc[dt]}).dropna()
            if len(rows) >= 5 and rows["f"].nunique() > 1 and rows["r"].nunique() > 1:
                ic.loc[dt, name] = rows["f"].rank().corr(rows["r"].rank())

    weights = pd.DataFrame(np.nan, index=fwd_ret.index, columns=names, dtype=float)
    for pos, dt in enumerate(fwd_ret.index):
        hist = ic.iloc[max(0, pos - window):pos]
        cnt = hist.count()
        mu = hist.mean(skipna=True)
        sig = hist.std(skipna=True).replace(0, np.nan)
        icir = mu.divide(sig).where(cnt >= min_periods)
        scores = icir.clip(lower=0).replace([np.inf, -np.inf], np.nan).dropna()
        if scores.sum() > 0:
            weights.loc[dt, scores.index] = scores / scores.sum()
        else:
            avail = [n for n, f in factors.items() if f.loc[dt].notna().any()]
            if avail:
                weights.loc[dt, avail] = 1.0 / len(avail)
    weights.index.name = Col.DATE
    return weights


def _stabilise_weights(
    w: pd.DataFrame,
    factors: dict[str, pd.DataFrame],
    min_pos: int,
) -> pd.DataFrame:
    """Ensure at least *min_pos* positive weights per period; otherwise
    fall back to equal weight over available factors."""
    out = w.copy()
    for dt in out.index:
        pos = out.loc[dt].dropna()
        pos = pos[pos > 0]
        if len(pos) >= min_pos:
            continue
        avail = [n for n, f in factors.items() if dt in f.index and f.loc[dt].notna().any()]
        out.loc[dt] = np.nan
        if len(avail) >= min_pos:
            out.loc[dt, avail] = 1.0 / len(avail)
    return out


def _weighted_combine(
    factors: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
    min_factors: int,
) -> pd.DataFrame:
    """Weighted average of factor values, only when ≥ *min_factors* are valid."""
    cols = next(iter(factors.values())).columns
    result = pd.DataFrame(np.nan, index=weights.index, columns=cols, dtype=float)
    for dt in weights.index:
        num = pd.Series(0.0, index=cols, dtype=float)
        den = pd.Series(0.0, index=cols, dtype=float)
        cnt = pd.Series(0, index=cols, dtype=int)
        for name, wt in weights.loc[dt].dropna().items():
            if wt <= 0 or name not in factors:
                continue
            row = factors[name].loc[dt].reindex(cols)
            mask = row.notna()
            num.loc[mask] += row.loc[mask] * wt
            den.loc[mask] += wt
            cnt.loc[mask] += 1
        valid = den.gt(0) & cnt.ge(min_factors)
        vc = valid.index[valid]
        result.loc[dt, vc] = (num.loc[vc] / den.loc[vc]).to_numpy()
    result.index.name = Col.DATE
    result.columns.name = Col.SYMBOL
    return result


def _compute_quality(
    proc: dict[str, pd.DataFrame],
    fwd: pd.DataFrame,
    window: int,
    min_periods: int,
    min_factors: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = _rolling_rank_icir_weights(proc, fwd, window, min_periods)
    w = _stabilise_weights(w, proc, min_factors)
    q = _weighted_combine(proc, w, min_factors)
    return q, w


def _compute_equal_weight_quality(
    proc: dict[str, pd.DataFrame],
    min_factors: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = next(iter(proc.values())).index
    names = list(proc)
    w = pd.DataFrame(np.nan, index=dates, columns=names, dtype=float)
    for dt in dates:
        avail = [n for n, f in proc.items() if f.loc[dt].notna().any()]
        if len(avail) >= min_factors:
            w.loc[dt, avail] = 1.0 / len(avail)
    w.index.name = Col.DATE
    q = _weighted_combine(proc, w, min_factors)
    return q, w


# ── Step 6: QualityIncrease ───────────────────────────────────────────── #

def _quality_increase(
    quality: pd.DataFrame,
    aligned: pd.DataFrame,
    lag_mode: str,
) -> pd.DataFrame:
    """QualityIncrease_t = Quality_t − Quality_{t-1}.

    *lag_mode* controls how t-1 is determined:

    - ``"yoy_report"`` — look up the Quality value that corresponded to
      last year's comparable report_date (preferred).
    - ``"4q"`` — simple 4-period shift on the signal-date grid.
    - ``"12m"`` — simple 12-period shift.
    """
    if lag_mode == "4q":
        return (quality - quality.shift(4)).replace([np.inf, -np.inf], np.nan)
    if lag_mode == "12m":
        return (quality - quality.shift(12)).replace([np.inf, -np.inf], np.nan)

    # yoy_report mode — match to last year's comparable report
    result = pd.DataFrame(np.nan, index=quality.index, columns=quality.columns, dtype=float)
    report_panel = aligned["report_date"].unstack(Col.SYMBOL).reindex_like(quality)
    yoy_report_panel = aligned["report_date_yoy"].unstack(Col.SYMBOL).reindex_like(quality)

    long_q = quality.stack(future_stack=True).rename("q").reset_index()
    long_r = report_panel.stack(future_stack=True).rename("rd").reset_index()
    q_by_rd = long_q.merge(long_r, on=[Col.DATE, Col.SYMBOL], how="left")
    q_by_rd = q_by_rd.dropna(subset=["q", "rd"])
    q_by_rd = q_by_rd.sort_values([Col.SYMBOL, "rd", Col.DATE])
    q_by_rd = q_by_rd.drop_duplicates([Col.SYMBOL, "rd"], keep="last")
    lookup = q_by_rd.set_index([Col.SYMBOL, "rd"])["q"]

    for dt in quality.index:
        for sym in quality.columns:
            cur = quality.loc[dt, sym]
            prior_rd = yoy_report_panel.loc[dt, sym]
            if pd.isna(cur) or pd.isna(prior_rd):
                continue
            key = (sym, pd.Timestamp(prior_rd))
            if key in lookup.index:
                result.loc[dt, sym] = cur - lookup.loc[key]

    # Fallback: if yoy_report yields nothing, use 4-quarter shift
    if result.dropna(how="all").empty:
        result = quality - quality.shift(4)

    result.index.name = Col.DATE
    result.columns.name = Col.SYMBOL
    return result.replace([np.inf, -np.inf], np.nan)


# ── Step 7: Final composite ────────────────────────────────────────────── #

def _final_quality_growth(
    quality: pd.DataFrame,
    q_inc: pd.DataFrame,
    fwd: pd.DataFrame,
    window: int,
    min_periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """zscore(Quality) and zscore(QualityIncrease), then rolling-RankICIR
    weighted combination → FinalFactor."""
    # Reuse neutralisation as zscore-only (pass trivial market-cap)
    z = _preprocess(
        {"Quality": quality, "QualityIncrease": q_inc},
        pd.DataFrame(1.0, index=quality.index, columns=quality.columns),
        None,
    )
    w = _rolling_rank_icir_weights(z, fwd, window, min_periods)
    final = _weighted_combine(z, w, min_factors=1)
    return final, w


# ── Build selected subfactors ──────────────────────────────────────────── #

def _build_subfactors(
    names: tuple[str, ...],
    data: dict[str, pd.DataFrame],
    stmt: pd.DataFrame,
    aligned: pd.DataFrame,
    mcap: pd.DataFrame,
    sig_dates: pd.DatetimeIndex,
    symbols: pd.Index,
) -> dict[str, pd.DataFrame]:
    subs: dict[str, pd.DataFrame] = {}
    if any(n in names for n in BASE_PROFITABILITY):
        subs.update(_base_profitability(aligned))
    if any(n in names for n in GROWTH_QUALITY):
        subs.update(_growth_quality(aligned))
    if "DividendRatioTTM" in names:
        subs["DividendRatioTTM"] = _dividend_ratio_ttm(data, stmt, mcap, sig_dates, symbols)
    return _reindex_factors({n: subs[n] for n in names}, sig_dates, symbols)


# ── Empty factor placeholder ──────────────────────────────────────────── #

def _empty(sig_dates: pd.DatetimeIndex, symbols: pd.Index) -> pd.DataFrame:
    f = pd.DataFrame(np.nan, index=sig_dates, columns=symbols, dtype=float)
    f.index.name = Col.DATE
    f.columns.name = Col.SYMBOL
    return f


# ── SP500 universe filter ─────────────────────────────────────────────── #

def _filter_sp500(
    symbols: pd.Index,
    sig_dates: pd.DatetimeIndex,
    data: dict[str, pd.DataFrame],
) -> pd.Index:
    """Point-in-time filter: for each signal date, keep only stocks that are
    S&P 500 constituents on that date.  Returns the *union* across all dates
    so the panel shape stays consistent (non-constituent cells become NaN).

    Falls back to *symbols* unchanged when ``data["sp500_constituents"]``
    is absent or empty.
    """
    sp500 = data.get("sp500_constituents")
    if sp500 is None or sp500.empty:
        return symbols

    sp500 = sp500.copy()
    if isinstance(sp500.index, pd.MultiIndex):
        sp500.index = pd.MultiIndex.from_arrays([
            pd.DatetimeIndex(pd.to_datetime(sp500.index.get_level_values(0))).tz_localize(None),
            sp500.index.get_level_values(1).astype(str),
        ], names=sp500.index.names)

    # Collect the union of constituents across all signal dates (backward fill)
    sp500_dates = sp500.index.get_level_values(0).unique().sort_values()
    all_constituents: set[str] = set()
    for dt in sig_dates:
        # Find latest available SP500 date on or before dt
        idx = sp500_dates.searchsorted(dt, side="right")
        if idx == 0:
            continue
        nearest = sp500_dates[idx - 1]
        try:
            syms_dt = sp500.loc[nearest].index.astype(str).tolist()
            all_constituents.update(syms_dt)
        except KeyError:
            continue

    if not all_constituents:
        return symbols

    filtered = symbols.intersection(sorted(all_constituents))
    logger.info(
        "quality_combine: SP500 filter applied, {} → {} symbols",
        len(symbols), len(filtered),
    )
    return filtered


# ── Main orchestrator ─────────────────────────────────────────────────── #

def _run(
    factor: QualityCombine,
    data: dict[str, pd.DataFrame],
    equal_subfactor_weights: bool = False,
    subfactor_names: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Full pipeline: statement → subfactors → preprocess → Quality →
    QualityIncrease → FinalQualityGrowth."""
    mkt = _norm_panel(data["market"])
    stmt = _prepare_statement(data.get("statement"))
    sig_d = _signal_dates(data, mkt)
    syms = mkt.index.get_level_values(Col.SYMBOL).unique()
    logger.info(
        "quality_combine: signal_dates={}, symbols={}",
        len(sig_d), len(syms),
    )

    # ── SP500 universe filter ──
    syms = _filter_sp500(syms, sig_d, data)

    mcap = _market_cap_panel(data, sig_d, syms)
    ind = _industry_panel(data, sig_d, syms)
    aligned = _align_latest(stmt, sig_d, syms)

    sel_names = subfactor_names or SUBFACTOR_NAMES
    subs = _build_subfactors(sel_names, data, stmt, aligned, mcap, sig_d, syms)
    for name, sf in subs.items():
        valid_per_date = sf.notna().sum(axis=1)
        logger.info(
            "  subfactor {}: dates_with_data={}, avg_valid_per_date={:.1f}/{}",
            name,
            int((valid_per_date >= factor.min_valid_cross_section).sum()),
            valid_per_date.mean(),
            len(syms),
        )
    final_factor = _empty(sig_d, syms)

    # ── Subfactor dates with enough cross-section ──
    ready_dates = _valid_dates(subs, factor.min_valid_cross_section)
    logger.info(
        "quality_combine: ready_dates after subfilter={}/{}",
        len(ready_dates), len(sig_d),
    )
    if ready_dates.empty:
        _save_debug(factor, aligned, subs, {}, final_factor, sig_d, sel_names)
        return final_factor

    fwd_full = _forward_returns(mkt, sig_d, syms)
    ready_subs = _reindex_factors(subs, ready_dates, syms)
    ready_mcap = mcap.reindex(index=ready_dates, columns=syms)
    ready_ind = ind.reindex(index=ready_dates, columns=syms) if ind is not None else None
    proc = _preprocess(ready_subs, ready_mcap, ready_ind)
    fwd = fwd_full.reindex(index=ready_dates, columns=syms)

    # ── Step 5: Quality ──
    if equal_subfactor_weights:
        quality, q_weights = _compute_equal_weight_quality(proc, factor.quality_min_factors)
    else:
        quality, q_weights = _compute_quality(
            proc, fwd, factor.rank_icir_window, factor.rank_icir_min_periods,
            factor.quality_min_factors,
        )

    # If only computing a subset (no QualityIncrease / Final step), return Quality directly
    if subfactor_names is not None and subfactor_names != SUBFACTOR_NAMES:
        final_factor.loc[quality.index, quality.columns] = quality
        _save_debug(factor, aligned, subs, proc, quality, sig_d, sel_names, q_weights=q_weights)
        final_factor.index.name = Col.DATE
        final_factor.columns.name = Col.SYMBOL
        return final_factor

    # ── Step 6: QualityIncrease ──
    q_inc = _quality_increase(quality, aligned, factor.quality_increase_lag)
    q_valid = quality.notna().sum(axis=1)
    qi_valid = q_inc.notna().sum(axis=1)
    logger.info(
        "quality_combine: Quality dates_with_data={}/{}, "
        "QualityIncrease dates_with_data={}/{}",
        int((q_valid >= factor.min_valid_cross_section).sum()), len(sig_d),
        int((qi_valid >= factor.min_valid_cross_section).sum()), len(sig_d),
    )

    # ── Step 7: FinalQualityGrowth ──
    final_inputs = {"Quality": quality, "QualityIncrease": q_inc}
    final_ready = _valid_dates(final_inputs, factor.min_valid_cross_section)
    logger.info(
        "quality_combine: final_ready_dates={}/{}",
        len(final_ready), len(sig_d),
    )
    if final_ready.empty:
        f_weights = pd.DataFrame(
            index=ready_dates, columns=["Quality", "QualityIncrease"], dtype=float,
        )
    else:
        r_q = quality.reindex(index=final_ready, columns=syms)
        r_qi = q_inc.reindex(index=final_ready, columns=syms)
        r_fwd = fwd.reindex(index=final_ready, columns=syms)
        scored, f_weights = _final_quality_growth(
            r_q, r_qi, r_fwd, factor.rank_icir_window, factor.rank_icir_min_periods,
        )
        final_factor.loc[scored.index, scored.columns] = scored

    # ── Debug introspection ──
    factor.last_statement = aligned
    factor.last_subfactors = subs
    factor.last_processed_subfactors = proc
    factor.last_quality = quality
    factor.last_quality_weights = q_weights
    factor.last_quality_increase = q_inc
    factor.last_final_weights = f_weights

    final_factor.index.name = Col.DATE
    final_factor.columns.name = Col.SYMBOL
    return final_factor


def _save_debug(
    factor, aligned, subs, proc, quality, sig_d, names, q_weights=None,
):
    """Persist intermediate results for inspection even on early-exit paths."""
    factor.last_statement = aligned
    factor.last_subfactors = subs
    factor.last_processed_subfactors = proc
    factor.last_quality = quality
    factor.last_quality_weights = q_weights if q_weights is not None else pd.DataFrame(
        index=sig_d, columns=list(names), dtype=float,
    )
    factor.last_quality_increase = quality.copy()
    factor.last_final_weights = pd.DataFrame(
        index=sig_d, columns=["Quality", "QualityIncrease"], dtype=float,
    )


# ══════════════════════════════════════════════════════════════════════════
#  Registered factor classes
# ══════════════════════════════════════════════════════════════════════════


@register_factor
class QualityCombine(BaseFactor):
    """Quality + Quality-Growth composite weighted by rolling RankICIR.

    Parameters (class attributes, overridable via __init__)
    ----------
    rank_icir_window : int
        Rolling window length (in signal-date periods) for RankIC/ICIR
        computation.  Default 24 ≈ 2 years of monthly signals.
    rank_icir_min_periods : int
        Minimum number of valid RankIC observations required before using
        ICIR-based weights (otherwise falls back to equal weight).
    quality_min_factors : int
        Minimum number of valid subfactors required to produce a Quality
        score for a given stock on a given date.
    quality_increase_lag : str
        How the t-1 reference point for QualityIncrease is chosen:

        - ``"yoy_report"`` — match to last year's comparable report_date
          (preferred, avoids seasonal noise).
        - ``"4q"`` — simple 4-period shift on the signal grid.
        - ``"12m"`` — simple 12-period shift.
    min_valid_cross_section : int
        Minimum number of stocks with valid data on a given date for that
        date to be included in the computation.
    """

    name = "quality_combine"
    description = "Quality + quality-increase composite weighted by rolling RankICIR"
    category = "quality"

    rank_icir_window: int = 24
    rank_icir_min_periods: int = 6
    quality_min_factors: int = 3
    quality_increase_lag: str = "yoy_report"
    min_valid_cross_section: int = 5

    # Debug introspection attributes (populated after generate_signals)
    last_statement: pd.DataFrame | None = None
    last_subfactors: dict | None = None
    last_processed_subfactors: dict | None = None
    last_quality: pd.DataFrame | None = None
    last_quality_weights: pd.DataFrame | None = None
    last_quality_increase: pd.DataFrame | None = None
    last_final_weights: pd.DataFrame | None = None

    def __init__(
        self,
        *,
        rank_icir_window: int | None = None,
        rank_icir_min_periods: int | None = None,
        quality_min_factors: int | None = None,
        quality_increase_lag: str | None = None,
        min_valid_cross_section: int | None = None,
    ) -> None:
        if rank_icir_window is not None:
            self.rank_icir_window = rank_icir_window
        if rank_icir_min_periods is not None:
            self.rank_icir_min_periods = rank_icir_min_periods
        if quality_min_factors is not None:
            self.quality_min_factors = quality_min_factors
        if quality_increase_lag is not None:
            if quality_increase_lag not in {"yoy_report", "4q", "12m"}:
                raise ValueError(
                    "quality_increase_lag must be one of: yoy_report, 4q, 12m"
                )
            self.quality_increase_lag = quality_increase_lag
        if min_valid_cross_section is not None:
            self.min_valid_cross_section = min_valid_cross_section

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return _run(self, data)


@register_factor
class QualityCombineEqualWeight(QualityCombine):
    """Same as :class:`QualityCombine` but with equal-weighted subfactors."""

    name = "quality_combine_equal_weight"
    description = "Quality + quality-increase composite with equal-weighted subfactors"

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return _run(self, data, equal_subfactor_weights=True)


@register_factor
class QualityCombineProfitability(QualityCombine):
    """Only the 4 base profitability subfactors (ROEQ/ROAQ/GPOAQ/GMARQ),
    combined via rolling RankICIR weighting."""

    name = "quality_combine_profitability"
    description = "Base profitability subfactors weighted by rolling RankICIR"

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return _run(self, data, subfactor_names=BASE_PROFITABILITY)


@register_factor
class QualityCombineGrowth(QualityCombine):
    """Only the 4 growth quality subfactors (dROEQ/dROAQ/dGPOAQ/dGMARQ),
    combined via rolling RankICIR weighting."""

    name = "quality_combine_growth"
    description = "Growth quality subfactors weighted by rolling RankICIR"

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return _run(self, data, subfactor_names=GROWTH_QUALITY)
