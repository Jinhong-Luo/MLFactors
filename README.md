# MLFactors

MLFactors 是一个基于本地美股日频数据的轻量因子研究项目。当前主流程从按股票拆分的 CSV 文件读取数据，计算已注册因子，并完成 IC、换手率、分层收益和风险指标评估，最终输出 CSV、图表和 Markdown 报告。

当前已实现的示例因子是 CAPM Beta：使用个股收益与市值加权市场组合收益的滚动协方差和方差计算 Beta。

## 当前流程

```text
cache/csv/*.csv
    -> dataloader.load_data()
    -> Runner.load()
    -> Factor.generate_signals()
    -> factors_eval.eval()
    -> FactorPlotter
    -> outputs/<factor>/
```

主要能力：

- 按股票读取 `cache/csv/<SYMBOL>.csv`
- 将数据组装为 `(date, symbol)` MultiIndex `DataFrame`
- 可选加载每日 S&P 500 成分股字典
- 通过简单注册表管理因子
- 默认评估 1、5、10、21 日前向收益
- 生成 IC、ICIR、换手率、分层收益、最大回撤和超额收益指标
- 在分层累计收益图中加入 SPY、QQQ 基准
- 可按股票保存完整回测期的因子值
- 生成包含汇总表和图片的 Markdown 报告

## 环境

优先使用项目根目录的虚拟环境：

```bash
./.venv/bin/python -c "import sys; print(sys.executable)"
```

当前主流程依赖：

- Python 3.10+
- pandas
- numpy
- scipy
- matplotlib
- tabulate，用于生成 Markdown 表格

可以用下面的命令检查核心依赖：

```bash
./.venv/bin/python -c "import pandas, numpy, scipy, matplotlib, tabulate"
```

项目直接从仓库根目录运行，不要求安装为 Python 包。

## 数据约定

### 股票 CSV

主流程默认读取：

```text
cache/csv/AAPL.csv
cache/csv/MSFT.csv
cache/csv/NVDA.csv
...
```

每个文件对应一只股票，至少包含：

| 字段 | 说明 |
| --- | --- |
| `date` | 交易日，格式为 `YYYY-MM-DD` |
| `close` | 个股收盘价，用于计算收益 |
| `market_cap` | 市值，用于构造市值加权市场组合 |
| `SPY_close` | SPY 收盘价，用于报告基准 |
| `QQQ_close` | QQQ 收盘价，用于报告基准 |

行情、估值、财务报表、宏观数据、行业和 ETF 等完整字段说明见 [meta_data/us_db_fields.md](meta_data/us_db_fields.md)。

CSV 数据遵循可用时间约束：

- 不包含生成数据时点之后的未来记录
- 财务报表从 `report_date + 45` 天起才允许进入交易日数据
- 宏观数据从其发布时间 `dt` 起才允许进入交易日数据
- 后续交易日只向前填充当时已经可用的数据

`cache/` 是本地数据目录，已被 `.gitignore` 忽略，不随代码仓库提交。

### S&P 500 成分股

[meta_data/sp500_constituents_daily.csv](meta_data/sp500_constituents_daily.csv) 的格式为：

```csv
date,tickers
2024-01-02,"AAPL,MSFT,NVDA,..."
```

传入该文件后，`load_data()` 返回的第二个对象是：

```python
{
    "2024-01-02": {"AAPL", "MSFT", "NVDA"},
}
```

不传 `constituents_path` 时返回空字典，因子使用本次加载的全部股票。

## 快速开始

### 运行内置示例

`run.py` 默认计算 AAPL、MSFT、NVDA 从 2024 年开始的 Beta，并将结果保存到 `outputs/beta/`：

```bash
./.venv/bin/python run.py
```

### 在代码中运行

因子模块必须先导入，注册表才能找到对应因子：

```python
import factors.beta  # 注册 beta 因子

from run import Runner


runner = Runner(
    factor_name="beta",
    factor_params={
        "lookback": 252,
        "min_obs": 120,
        "clip": (-3.0, 3.0),
    },
    symbols=None,  # None 表示读取 cache/csv 下的全部股票
    start="2017-01-01",
    end=None,
    constituents_path=None,
    data_columns=["close", "market_cap"],
    forward_periods=(1, 5, 10, 21),
    n_groups=5,
    output_dir="outputs/beta_all_2017_latest",
)

runner.run(save_factor=True)
print(runner.summary)
```

如需仅使用当日 S&P 500 成分股，可设置：

```python
constituents_path="meta_data/sp500_constituents_daily.csv"
```

只加载因子实际使用的列可以显著减少全市场运行时的内存占用。

## Beta 因子

`factors/beta.py` 中的 Beta 使用解析式：

```text
beta_i = cov(stock_return_i, market_return) / var(market_return)
```

其中市场收益由当前股票池内个股收益按 `market_cap` 横截面加权得到。默认参数：

- `lookback=252`：滚动窗口为 252 个交易日
- `min_obs=120`：有效样本少于 120 时输出空值
- `clip=(-3, 3)`：将最终 Beta 限制在该区间

市场收益方差为零、方差缺失或有效样本不足时，对应 Beta 为 `NaN`。

## 评估口径

`Runner` 默认对 1、5、10、21 日周期分别评估：

- Rank IC 均值、标准差、ICIR、t 统计量和 p 值
- IC 为正的比例
- 最高因子组换手率
- 分层累计收益和各组年化收益
- 最高组多头夏普和最大回撤
- 最低组反向持有的最大回撤
- 最高组相对各组等权基准的年化超额收益、最大回撤和 Calmar
- IC 衰减曲线

前向收益按以下方式计算：

```text
price[t + 1 + period] / price[t + 1] - 1
```

即因子形成后的下一个交易日进入，并在持有 `period` 个交易日后退出。对于多日前向收益，评估函数每隔 `period` 个交易日采样一次，避免把每日重叠的多日收益连续复利。

## 输出结果

`runner.run(save_factor=True)` 会生成：

```text
outputs/<factor>/
├── report.md                 # 配置、汇总表和图片链接
├── factor_summary.csv        # 多周期汇总指标
├── <factor>_1d.png           # 1 日评估图
├── <factor>_5d.png           # 5 日评估图
├── <factor>_10d.png          # 10 日评估图
├── <factor>_21d.png          # 21 日评估图
└── factor/
    ├── AAPL.csv              # date、因子名两列
    ├── MSFT.csv
    └── ...
```

每张评估图包含 IC 时序、IC 分布、分层累计收益、IC 衰减、换手率和指标表。分层累计收益子图同时显示 SPY、QQQ 的累计收益。

当前工作区已经完成一次 2017 年起的全股票 Beta 回测，报告位于 `outputs/beta_all_2017_latest/report.md`。该目录属于本地输出，不随 Git 提交。

## 自定义因子

新因子只需要继承 `BaseFactor`、定义 `name`，并使用 `register_factor` 注册。`generate_signals()` 应返回以日期为索引、股票代码为列的宽表：

```python
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class Momentum20(BaseFactor):
    name = "momentum_20"
    description = "20 日价格动量"

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        close = data["close"].unstack("symbol")
        return close.pct_change(20)
```

将上述实现保存为 `factors/momentum.py`。使用前先导入因子模块，然后交给通用 `Runner`：

```python
import factors.momentum

from run import Runner


runner = Runner(
    factor_name="momentum_20",
    data_columns=["close"],
)
runner.run()
```

注册表提供最小的注册、获取和列表功能：

```python
from factors.registry import FactorRegistry

print(FactorRegistry.list())
factor_class = FactorRegistry.get("beta")
```

## 项目结构

```text
MLFactors/
├── cache/
│   ├── us.db                         # 本地原始数据库
│   └── csv/                          # 按股票拆分的数据文件
├── factors/
│   ├── base.py                       # 因子抽象基类
│   ├── registry.py                   # 简单因子注册表
│   └── beta.py                       # CAPM Beta 因子
├── lab/                              # 数据维护和研究脚本，不属于主运行接口
├── meta_data/
│   ├── us_db_fields.md               # 数据字段说明
│   ├── sp500_constituents_daily.csv  # 每日 S&P 500 成分股
│   ├── sp500_sectors.csv             # S&P 500 行业信息
│   └── RUSSELL1000.txt                # Russell 1000 股票列表
├── dataloader.py                     # CSV 和成分股加载
├── factors_eval.py                   # 因子评估指标
├── plot.py                           # 六子图评估图
└── run.py                            # 通用 Runner
```

## 注意事项

- `symbols=None` 会读取 `cache/csv/` 下全部股票，数据量较大。
- Beta 的市场组合取决于实际加载的股票池；只加载少量股票时，它不是完整市场组合。
- 不传成分股文件时，全区间使用当前 CSV 股票集合，可能存在幸存者偏差。
- 因子值越高对应分层编号越高，但这不代表所有因子都应当做多最高组；需要结合因子定义解释方向。
- 当前实现用于本地研究和流程验证，不包含交易成本、滑点、成交量约束和真实撮合。

## License

MIT
