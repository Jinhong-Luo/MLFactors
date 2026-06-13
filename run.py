"""加载 CSV 数据并计算注册因子。"""

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from dataloader import load_data
from factors_eval import FactorEvalResult, eval as evaluate_factor
from factors.registry import FactorRegistry
from plot import FactorPlotter


class Runner:
    """管理因子的数据加载、计算和结果查看。"""

    def __init__(
        self,
        factor_name: str,
        factor_params: dict[str, Any] | None = None,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
        constituents_path: str | Path | None = None,
        data_columns: list[str] | None = None,
        forward_periods: tuple[int, ...] = (1, 5, 10, 21),
        n_groups: int = 5,
        ic_method: str = "rank",
        max_lag: int = 20,
        output_dir: str | Path | None = None,
    ) -> None:
        self.factor_name = factor_name
        self.symbols = symbols
        self.start = start
        self.end = end
        self.constituents_path = constituents_path
        self.data_columns = data_columns
        self.forward_periods = forward_periods
        self.n_groups = n_groups
        self.ic_method = ic_method
        self.max_lag = max_lag
        self.output_dir = Path(output_dir or Path("outputs") / factor_name)
        self.factor = FactorRegistry.get(factor_name)(**(factor_params or {}))
        self.data = pd.DataFrame()
        self.benchmark = pd.DataFrame()
        self.constituents: dict[str, set[str]] = {}
        self.result = pd.DataFrame()
        self.evaluations: dict[int, FactorEvalResult] = {}
        self.summary = pd.DataFrame()

    def load(self) -> pd.DataFrame:
        self.data, self.constituents = load_data(
            symbols=self.symbols,
            start=self.start,
            end=self.end,
            constituents_path=self.constituents_path,
            columns=self.data_columns,
        )
        return self.data

    def load_benchmark(self) -> pd.DataFrame:
        """提取回测期间的 SPY 和 QQQ 收盘价。"""
        source_symbol = self.data.index.get_level_values("symbol")[0]
        benchmark_data, _ = load_data(
            symbols=[source_symbol],
            start=self.start,
            end=self.end,
            columns=["SPY_close", "QQQ_close"],
        )
        self.benchmark = benchmark_data.xs(source_symbol, level="symbol").rename(
            columns={"SPY_close": "SPY", "QQQ_close": "QQQ"}
        )[["SPY", "QQQ"]]
        return self.benchmark

    def calculate(self, save: bool = False) -> pd.DataFrame:
        """计算因子；save=True 时按股票保存完整回测期因子值。"""
        self.result = self.factor.generate_signals(self.data, self.constituents)
        if save:
            factor_dir = self.output_dir / "factor"
            factor_dir.mkdir(parents=True, exist_ok=True)
            symbols = self.data.index.get_level_values("symbol").unique()
            factor_result = self.result.reindex(columns=symbols)
            for symbol in symbols:
                dates = self.data.xs(symbol, level="symbol").index
                factor_data = factor_result[symbol].reindex(dates).rename(self.factor_name)
                factor_data.index = factor_data.index.strftime("%Y-%m-%d")
                factor_data.index.name = "date"
                factor_data.to_csv(
                    factor_dir / f"{symbol}.csv",
                    na_rep="",
                )
        return self.result

    def evaluate(self) -> dict[int, FactorEvalResult]:
        """计算 1、5、10、21 日等指定周期的因子评估结果。"""
        self.evaluations = {
            period: evaluate_factor(
                self.result,
                self.data,
                forward_period=period,
                n_groups=self.n_groups,
                ic_method=self.ic_method,
                max_lag=self.max_lag,
            )
            for period in self.forward_periods
        }
        self.summary = pd.concat(
            [evaluation.summary for evaluation in self.evaluations.values()]
        ).sort_index()
        return self.evaluations

    def save_reports(self) -> Path:
        """保存 CSV、综合评估图和包含表格与图片的 Markdown 报告。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary.to_csv(self.output_dir / "factor_summary.csv")

        image_files = []
        for period, evaluation in self.evaluations.items():
            output_path = self.output_dir / f"{self.factor_name}_{period}d.png"
            benchmark_returns = (
                self.benchmark.shift(-(1 + period))
                / self.benchmark.shift(-1)
                - 1
            )
            benchmark_returns = benchmark_returns.reindex(
                evaluation.layered.group_returns.index
            )
            benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
            FactorPlotter(
                evaluation,
                factor_name=self.factor_name,
                benchmark_cumulative=benchmark_cumulative,
            ).save(output_path)
            image_files.append(output_path)

        report_lines = [
            f"# {self.factor_name} 因子评估报告",
            "",
            "## 运行配置",
            "",
            f"- 数据区间：{self.data.index.get_level_values('date').min().date()} 至 "
            f"{self.data.index.get_level_values('date').max().date()}",
            f"- 股票数量：{self.data.index.get_level_values('symbol').nunique()}",
            f"- 前向收益周期：{', '.join(f'{period} 日' for period in self.forward_periods)}",
            f"- 分层数量：{self.n_groups}",
            f"- IC 方法：{self.ic_method}",
            "",
            "## 多周期评估汇总",
            "",
            self.summary.reset_index().to_markdown(index=False),
            "",
            "## 评估图表",
            "",
        ]
        for period, image_file in zip(self.evaluations, image_files):
            report_lines.extend([
                f"### {period} 日前向收益",
                "",
                f"![{self.factor_name} {period} 日评估图]({image_file.name})",
                "",
            ])

        report_path = self.output_dir / "report.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        return report_path

    def latest(self) -> pd.Series:
        """返回最近一个有因子结果的交易日。"""
        latest_result = self.result.dropna(how="all")
        if latest_result.empty:
            return pd.Series(dtype=float, name=self.factor_name)
        result = latest_result.iloc[-1].dropna().sort_values()
        result.name = latest_result.index[-1]
        return result

    def run(self, save_factor: bool = False) -> dict[int, FactorEvalResult]:
        """依次加载数据、计算因子、执行多周期评估并保存结果。"""
        self.load()
        self.load_benchmark()
        self.calculate(save=save_factor)
        self.evaluate()
        self.save_reports()
        return self.evaluations


if __name__ == "__main__":
    import factors.beta  # noqa: F401

    runner = Runner(
        factor_name="beta",
        factor_params={"lookback": 252, "min_obs": 120, "clip": (-3.0, 3.0)},
        symbols=["AAPL", "MSFT", "NVDA"],
        start="2024-01-01",
        data_columns=["close", "market_cap"],
        n_groups=3,
        output_dir="outputs/beta",
    )
    runner.run(save_factor=True)
    latest_result = runner.latest()
    print(f"最新 {runner.factor_name} 结果 ({latest_result.name.date()}):")
    print(latest_result.to_string())
    print("\n多周期评估汇总:")
    print(runner.summary.to_string())
    print(f"\n结果已保存到: {runner.output_dir.resolve()}")
