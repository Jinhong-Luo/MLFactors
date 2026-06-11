"""
美股调仓周期因子测试脚本（本地 SQLite 数据）
============================================
基于本地 ``cache/us.db``，因子接收原始日频行情，并按自然周、自然半月
或自然月的调仓日期生成信号；评估时用同一调仓周期聚合行情计算单周期
前向收益。

用法：
  ./.venv/bin/python run_us_rebalance_factor_test.py

常用参数：
  --rebalance       调仓周期：week / half_month / month
  --factors         因子名称，默认 vff3
  --db-path         SQLite 数据库路径
  --start / --end   回测区间
  --output-dir      报告输出目录
  --no-plots        只输出汇总 CSV，不保存图表
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from scipy import stats

from data.local_loader import USStockLocalLoader
from data.schema import Col
from evaluation.plot import plot_factor_report
from evaluation.selection.ic import calc_icir, calc_t_stat
from evaluation.selection.layered import layered_backtest
from factors.registry import FactorRegistry
from pipeline.selection_runner import SelectionPipeline


DEFAULT_FACTORS = ["quality_combine_equal_weight"]
DEFAULT_DB_PATH = ROOT / "cache/us.db"
DEFAULT_START_DATE = "2018-01-04"
DEFAULT_END_DATE = "2026-05-28"
DEFAULT_REBALANCE = "week"
DEFAULT_OUTPUT_DIR = Path("outputs/us_rebalance_quality_combine_equal_weight_week_fixIndustry_sp500")
REBALANCE_LABELS = {
    "week": "1w",
    "half_month": "1half_month",
    "month": "1m",
}
REBALANCE_PERIODS_PER_YEAR = {
    "week": 52,
    "half_month": 24,
    "month": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行美股调仓周期选股因子测试")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--market", default="US")
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--end", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--rebalance",
        choices=["week", "half_month", "month"],
        default=DEFAULT_REBALANCE,
        help="按自然周期生成信号日期，并测试 1 个聚合周期的前向收益",
    )
    parser.add_argument("--factors", nargs="+", default=DEFAULT_FACTORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def get_market_symbols(
    db_path: Path,
    start: str | None,
    end: str | None,
) -> list[str]:
    """只根据 ``market`` 表确定股票池。"""
    if not db_path.exists():
        logger.error("SQLite 数据库不存在: {}", db_path)
        return []

    conditions: list[str] = []
    params: list[object] = []
    if start:
        conditions.append("date(dt) >= ?")
        params.append(str(pd.Timestamp(start).date()))
    if end:
        conditions.append("date(dt) <= ?")
        params.append(str(pd.Timestamp(end).date()))

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"SELECT DISTINCT symbol FROM market {where} ORDER BY symbol"
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(query, params).fetchall()
    symbols = [row[0] for row in rows]
    logger.info("从 market 表加载 {} 只股票（区间 {} ~ {}）", len(symbols), start, end)
    return symbols


def resolve_factor_names(factors: list[str]) -> list[str]:
    """展开内置因子集合别名，保持普通因子名原样。"""
    resolved: list[str] = []
    for name in factors:
        if name in {"alpha158", "alpha158_all"}:
            resolved.extend(
                factor_name
                for factor_name in FactorRegistry.list()
                if factor_name.startswith("alpha158_")
            )
        else:
            resolved.append(name)
    return sorted(dict.fromkeys(resolved))


def benchmark_forward_cumulative_returns(
    market_data: pd.DataFrame,
    index: pd.Index,
    symbols: tuple[str, ...] = ("SPY", "QQQ"),
) -> pd.DataFrame:
    """按同一聚合周期的下一期收益计算基准累计收益。"""
    if market_data.empty or Col.CLOSE not in market_data.columns:
        return pd.DataFrame()

    close = market_data[Col.CLOSE].unstack(Col.SYMBOL).sort_index()
    present_symbols = [symbol for symbol in symbols if symbol in close.columns]
    if not present_symbols:
        return pd.DataFrame()

    forward_returns = close[present_symbols].shift(-2) / close[present_symbols].shift(-1) - 1
    forward_returns = forward_returns.reindex(index)
    cumulative = (1.0 + forward_returns.fillna(0.0)).cumprod() - 1.0
    cumulative.index.name = Col.DATE
    return cumulative


def build_yearly_ic_summary(
    reports: dict,
    period_label: str,
    rebalance: str,
    periods_per_year: int,
) -> pd.DataFrame:
    records = []
    for factor_name, report in reports.items():
        ic_series = report.ic_series().dropna()
        if ic_series.empty:
            continue

        for year, yearly_ic in ic_series.groupby(ic_series.index.year):
            if len(yearly_ic) < 2:
                continue

            t_stat, _ = calc_t_stat(yearly_ic)
            df = len(yearly_ic) - 1
            p_value = stats.t.sf(abs(t_stat), df) * 2 if df > 0 else float("nan")
            records.append(
                {
                    "factor": factor_name,
                    "year": int(year),
                    "period_label": period_label,
                    "rebalance": rebalance,
                    "period_bars": 1,
                    "n": int(len(yearly_ic)),
                    "IC_mean": round(float(yearly_ic.mean()), 6),
                    "IC_std": round(float(yearly_ic.std()), 6),
                    "ICIR": round(
                        float(calc_icir(yearly_ic, 1, periods_per_year=periods_per_year, annualize=True)),
                        6,
                    ),
                    "t_stat": round(float(t_stat), 6),
                    "p_value": round(float(p_value), 6),
                    "IC_positive_ratio": round(float((yearly_ic > 0).mean()), 6),
                }
            )

    yearly_ic = pd.DataFrame.from_records(records)
    if not yearly_ic.empty:
        yearly_ic = yearly_ic.sort_values(["factor", "year", "period_bars"]).reset_index(drop=True)
    return yearly_ic


def adjust_reports_for_rebalance(reports: dict, periods_per_year: int) -> None:
    """用调仓周期对应的年化周期数重算分层回测缓存。"""
    for report in reports.values():
        report._layered = layered_backtest(
            report.factor_values,
            report.fwd_returns,
            report.n_groups,
            annual_trading_days=periods_per_year,
            period=1,
        )
        report._summary = None


def summarize_report(report, periods_per_year: int) -> dict[str, float]:
    ic_s = report.ic_series()
    t_stat, _ = calc_t_stat(ic_s)
    df = len(ic_s.dropna()) - 1
    p_value = stats.t.sf(abs(t_stat), df) * 2 if df > 0 else float("nan")
    layered = report.layered()
    long_sharpe = 0.0
    if not layered.sharpe_ratios.empty:
        long_sharpe = float(layered.sharpe_ratios.loc[layered.sharpe_ratios.index.max()])

    return {
        "period_bars": 1,
        "IC_mean": round(float(ic_s.mean()), 4),
        "IC_std": round(float(ic_s.std()), 4),
        "ICIR": round(float(calc_icir(ic_s, 1, periods_per_year=periods_per_year, annualize=True)), 4),
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "IC>0_ratio": round(float((ic_s > 0).mean()), 4),
        "turnover": round(float(report.turnover().mean()), 4),
        "long_sharpe": round(long_sharpe, 4),
        "long_max_drawdown": round(float(layered.long_max_drawdown), 4),
        "short_max_drawdown": round(float(layered.short_max_drawdown), 4),
        "top_excess_annual": round(float(layered.top_excess_annual), 4),
        "top_excess_max_dd": round(float(layered.top_excess_max_drawdown), 4),
        "top_excess_calmar": round(float(layered.top_excess_calmar), 4),
    }


def _format_markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "无数据\n"

    table = df.reset_index(drop=True).copy()
    columns = [str(col) for col in table.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_format_markdown_value(row[col]) for col in table.columns) + " |")
    return "\n".join(lines) + "\n"


def save_markdown_report(
    reports: dict,
    output_dir: Path,
    summary: pd.DataFrame,
    yearly_ic: pd.DataFrame,
    plot_paths: dict[str, Path],
    period_label: str,
    rebalance: str,
) -> None:
    report_path = output_dir / "report.md"
    lines = [
        "# 美股调仓周期因子报告",
        "",
        f"- 调仓周期: `{rebalance}`",
        f"- 收益周期: `{period_label}`",
        f"- 因子数: {len(reports)}",
        "",
        "## 汇总结果",
        "",
        dataframe_to_markdown(summary),
        "## 分年度 IC",
        "",
        dataframe_to_markdown(yearly_ic),
        "## 图表",
        "",
    ]

    if plot_paths:
        for factor_name, plot_path in plot_paths.items():
            lines.extend([
                f"### {factor_name}",
                "",
                f"![{factor_name}]({plot_path.name})",
                "",
            ])
    else:
        lines.extend(["未生成图表。", ""])

    lines.extend(["## 每次调仓股票列表", ""])
    for factor_name, report in reports.items():
        holdings = report.rebalance_holdings().copy()
        if not holdings.empty:
            holdings.insert(0, "factor", factor_name)
        lines.extend([
            f"### {factor_name}",
            "",
            dataframe_to_markdown(holdings),
        ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown 报告已保存: {}", report_path)


def save_reports(
    reports: dict,
    output_dir: Path,
    save_plots: bool,
    period_label: str,
    rebalance: str,
    periods_per_year: int,
    benchmark_data: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    plot_paths: dict[str, Path] = {}
    for name, report in reports.items():
        summary_df = pd.DataFrame([summarize_report(report, periods_per_year)])
        summary_df.insert(0, "factor", name)
        summary_df.insert(1, "period_label", period_label)
        summary_df.insert(2, "rebalance", rebalance)
        rows.append(summary_df)

        if save_plots:
            layered = report.layered()
            benchmark_cumulative = benchmark_forward_cumulative_returns(
                benchmark_data if benchmark_data is not None else report.market_data,
                index=layered.cumulative_returns.index,
            )
            fig = plot_factor_report(
                report.ic_series(),
                layered,
                report.ic_decay(),
                benchmark_cumulative=benchmark_cumulative,
                factor_name=name,
                period=1,
                period_label=period_label,
            )
            save_path = output_dir / f"{name}_{period_label}_report.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            plot_paths[name] = save_path
            logger.info("图表已保存: {}", save_path)

    summary = pd.concat(rows)
    summary_path = output_dir / "factor_summary.csv"
    summary.to_csv(summary_path)
    logger.info("汇总表已保存: {}", summary_path)

    yearly_ic = build_yearly_ic_summary(reports, period_label, rebalance, periods_per_year)
    yearly_ic_path = output_dir / "yearly_ic.csv"
    yearly_ic.to_csv(yearly_ic_path, index=False)
    logger.info("分年度 IC 表已保存: {}", yearly_ic_path)

    save_markdown_report(
        reports=reports,
        output_dir=output_dir,
        summary=summary,
        yearly_ic=yearly_ic,
        plot_paths=plot_paths,
        period_label=period_label,
        rebalance=rebalance,
    )

    print("\n" + "=" * 55)
    print("  美股调仓周期因子汇总")
    print("=" * 55)
    print(summary.to_string())


def main() -> int:
    args = parse_args()
    period_label = REBALANCE_LABELS[args.rebalance]
    periods_per_year = REBALANCE_PERIODS_PER_YEAR[args.rebalance]
    factor_names = resolve_factor_names(args.factors)
    logger.info("本次评估因子数: {}", len(factor_names))

    symbols = get_market_symbols(args.db_path, args.start, args.end)
    if not symbols:
        logger.error("股票列表为空，退出")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_symbols_path = args.output_dir / "selected_symbols.csv"
    pd.Series(symbols, name="symbol").to_csv(selected_symbols_path, index=False)
    logger.info("股票列表已保存: {}", selected_symbols_path)

    loader = USStockLocalLoader(db_path=args.db_path, market=args.market)
    pipeline = (
        SelectionPipeline(rebalance=args.rebalance)
        .set_data_loader(loader)
        .add_factors(factor_names)
    )

    reports = pipeline.run(
        symbols=symbols,
        start=args.start,
        end=args.end,
        show_plot=False,
    )
    adjust_reports_for_rebalance(reports, periods_per_year)

    benchmark_data = SelectionPipeline.aggregate_market_data(
        loader.load_etf_market_data(["SPY", "QQQ"], args.start, args.end),
        args.rebalance,
    )
    save_reports(
        reports=reports,
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
        period_label=period_label,
        rebalance=args.rebalance,
        periods_per_year=periods_per_year,
        benchmark_data=benchmark_data,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
