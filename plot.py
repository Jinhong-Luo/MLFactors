"""因子评估结果可视化。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factors_eval import FactorEvalResult


class FactorPlotter:
    """绘制 ``eval()`` 返回的完整因子评估结果。"""

    def __init__(
        self,
        result: FactorEvalResult,
        factor_name: str = "Factor",
        benchmark_cumulative: pd.DataFrame | None = None,
    ) -> None:
        self.result = result
        self.factor_name = factor_name
        self.benchmark_cumulative = benchmark_cumulative

    def plot_ic_series(self, ax: plt.Axes, rolling_window: int = 20) -> None:
        """绘制 IC 时间序列及滚动均值。"""
        ic = self.result.ic_series.dropna()
        ax.set_title("IC Time Series")
        if ic.empty:
            return
        ax.fill_between(
            ic.index, 0, ic.values,
            where=ic.values >= 0,
            color="#2c7bb6",
            alpha=0.35,
            label="IC > 0",
        )
        ax.fill_between(
            ic.index, 0, ic.values,
            where=ic.values < 0,
            color="#d7191c",
            alpha=0.35,
            label="IC < 0",
        )
        rolling_mean = ic.rolling(rolling_window).mean()
        ax.plot(rolling_mean.index, rolling_mean, color="black", label=f"MA({rolling_window})")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel("IC")
        ax.legend(fontsize=8)

    def plot_ic_histogram(self, ax: plt.Axes, bins: int = 50) -> None:
        """绘制 IC 分布直方图。"""
        ic = self.result.ic_series.dropna()
        ax.set_title("IC Distribution")
        if ic.empty:
            return
        ax.hist(ic, bins=bins, color="steelblue", alpha=0.75, edgecolor="white")
        ax.axvline(ic.mean(), color="red", linestyle="--", label=f"Mean={ic.mean():.4f}")
        ax.set_xlabel("IC")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    def plot_layered_returns(self, ax: plt.Axes) -> None:
        """绘制各因子分组的累计收益。"""
        cumulative = self.result.layered.cumulative_returns
        ax.set_title("Layered Cumulative Returns")
        if cumulative.empty:
            return
        colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(cumulative.columns)))
        for color, group in zip(colors, cumulative.columns):
            ax.plot(cumulative.index, cumulative[group], color=color, label=f"Group {group}")
        if self.benchmark_cumulative is not None:
            benchmark_colors = {"SPY": "black", "QQQ": "dimgray"}
            for benchmark in self.benchmark_cumulative.columns:
                ax.plot(
                    self.benchmark_cumulative.index,
                    self.benchmark_cumulative[benchmark],
                    color=benchmark_colors.get(benchmark, "black"),
                    linestyle="--",
                    linewidth=1.5,
                    label=benchmark,
                )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Cumulative Return")
        ax.legend(fontsize=8)

    def plot_ic_decay(self, ax: plt.Axes) -> None:
        """绘制不同滞后期的平均 IC。"""
        decay = self.result.ic_decay.dropna()
        ax.set_title("IC Decay")
        if decay.empty:
            return
        ax.bar(decay.index, decay.values, color="steelblue", alpha=0.75)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Mean IC")

    def plot_turnover(self, ax: plt.Axes) -> None:
        """绘制最高因子分组组合的换手率。"""
        turnover = self.result.turnover.dropna()
        ax.set_title("Portfolio Turnover")
        if turnover.empty:
            return
        ax.plot(turnover.index, turnover.values, color="darkorange", linewidth=1.0)
        ax.axhline(turnover.mean(), color="black", linestyle="--", label=f"Mean={turnover.mean():.4f}")
        ax.set_ylabel("Turnover")
        ax.legend(fontsize=8)

    def plot_summary(self, ax: plt.Axes) -> None:
        """以表格形式展示核心汇总指标。"""
        summary = self.result.summary.reset_index()
        formatted = summary.copy()
        for column in formatted.columns:
            if column != "period":
                formatted[column] = formatted[column].map(
                    lambda value: "NaN" if not np.isfinite(value) else f"{value:.4f}"
                )
        ax.axis("off")
        ax.set_title("Summary")
        table = ax.table(
            cellText=formatted.T.values,
            rowLabels=formatted.columns,
            colLabels=["Value"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.2)

    def plot(self, figsize: tuple[float, float] = (16, 14)) -> plt.Figure:
        """生成包含全部评估图表的综合报告图。"""
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        period = self.result.summary.index[0]
        fig.suptitle(
            f"Factor Report: {self.factor_name} | Forward Period: {period}d",
            fontsize=15,
            fontweight="bold",
        )
        self.plot_ic_series(axes[0, 0])
        self.plot_ic_histogram(axes[0, 1])
        self.plot_layered_returns(axes[1, 0])
        self.plot_ic_decay(axes[1, 1])
        self.plot_turnover(axes[2, 0])
        self.plot_summary(axes[2, 1])
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    def save(self, output_path: str | Path, dpi: int = 150) -> Path:
        """生成综合报告图并保存到指定路径。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure = self.plot()
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return output_path
