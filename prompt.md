请根据以下要求编写一个用于复现《人工智能选股之支持向量机模型》中“高斯核 SVM 训练阶段”的 Python 脚本。重点只实现训练阶段，不实现完整回测。要求代码结构清晰、函数名固定、输入输出提前声明、避免未来函数，并尽量使用 pandas + numpy + scikit-learn。

# 一、目标

实现一个高斯核 SVM 选股模型的训练流程：

```text
T 月月末股票因子暴露 X
        ↓
特征预处理
        ↓
根据 T+1 月超额收益构造分类标签 y
        ↓
样本合并
        ↓
训练集 / 交叉验证集划分
        ↓
网格搜索 C 和 gamma
        ↓
选择交叉验证集 AUC 最高的高斯核 SVM
        ↓
输出训练好的模型、PCA、最优参数、交叉验证结果
```

模型拟合目标不是下月具体收益率，而是二分类标签：

```text
若股票 T+1 月超额收益位于当月截面前 30%，则 y = 1
若股票 T+1 月超额收益位于当月截面后 30%，则 y = -1
中间 40% 样本不参与 SVM 训练
```

# 二、输入数据约定

核心函数请假设外部已经提供以下数据，不需要编写新的数据下载逻辑。真实项目入口需要优先复用当前仓库已有的数据加载、调仓日期、因子面板和收益计算能力。

## 1. factor_panel

```python
factor_panel: pd.DataFrame
```

格式：

```text
index: MultiIndex[date, symbol]
columns: factor_cols
values: 每只股票在每个月末的原始因子暴露
```

要求：

* `date` 是月末交易日。
* `symbol` 是股票代码。
* `factor_cols` 是因子名称列表，理论上可以是文章中的 70 个因子，也可以是用户实际传入的任意因子列表。
* 每一行表示某个截面期某只股票的原始因子暴露。
* 在真实 Alpha158 数据路径中，`factor_panel` 的构造可以参考 `lab/build_alpha158_rebalance_report.py` 的 `load_alpha158_on_signal_dates(db_path, signal_dates, factor_names, symbols)`，直接读取调仓信号日上的 Alpha158 因子暴露。

## 2. close

```python
close: pd.DataFrame | None
```

格式：

```text
index: trading_date
columns: symbol
values: 股票收盘价
```

用于在未提供 `market_data` / `signal_dates` 时，fallback 计算从当前月末到下个月末的个股收益。

## 3. benchmark_close

```python
benchmark_close: pd.Series
```

格式：

```text
index: trading_date
values: 基准指数收盘价
```

用于计算下月基准收益，进而得到个股下月超额收益。
真实项目路径中使用 SPY500 ETF 收盘价序列即可。

## 4. market_cap

```python
market_cap: pd.DataFrame
```

格式：

```text
index: date
columns: symbol
values: 股票总市值或流通市值
```

用于市值中性化。中性化时使用 `np.log(market_cap)`。

## 5. 现有框架复用约定

训练脚本需要与当前项目框架适配，以复用已有功能：

* 优先使用 `pipeline/selection_runner.py` 中的 `SelectionPipeline.load_data()` 获取 `market` 数据和已有的 `signal_dates`。当使用月度调仓时，可以通过 `SelectionPipeline(rebalance="month")` 或现有聚合逻辑得到月度信号日。
* `get_month_end_dates` 和 `align_month_end_panel` 保留为纯函数 fallback；真实项目路径中应优先使用 `SelectionPipeline.load_data()` 产生的数据日期和信号日，避免重复实现一套独立加载逻辑。
* `compute_forward_returns` 在拿到标准 `market_data: pd.DataFrame` 和 `signal_dates: pd.DatetimeIndex` 时，应优先复用 `evaluation/selection/report.py` 的 `calc_factors_returns(market_data, signal_dates)`；该函数返回 MultiIndex[date, symbol] 的前向收益 Series，可按需要 `unstack` 成 date x symbol。
* `compute_forward_excess_returns` 的 `benchmark_close` 使用 SPY500 ETF 数据即可，例如本地数据中代表 S&P 500 ETF 的 `SPY` 收盘价序列；不要用股票池等权收益替代基准。
* Alpha158 训练样本的 `factor_panel` 可以参考 `lab/build_alpha158_rebalance_report.py` 的 `load_alpha158_on_signal_dates` 从 `alpha158` 表按 `signal_dates` 和 `symbols` 读取。
* 如果现有框架数据不可用，才使用本文档中基于 `close`、`benchmark_close`、`factor_panel` 的模拟数据 fallback，以保证脚本最小示例可运行。

# 三、输出结果

请定义以下 dataclass：

```python
from dataclasses import dataclass
from typing import Any
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA

@dataclass
class SVMTrainingResult:
    model: SVC
    pca: PCA | None
    best_params: dict[str, float]
    cv_results: pd.DataFrame
    train_index: pd.MultiIndex
    valid_index: pd.MultiIndex
    feature_columns: list[str]
    preprocessing_summary: dict[str, Any]
```

其中：

* `model`：最终训练好的高斯核 SVM 模型。
* `pca`：训练集上拟合的 PCA；如果配置关闭 PCA，则为 `None`。
* `best_params`：最优参数，例如 `{"C": 10.0, "gamma": 0.01}`。
* `cv_results`：每组参数在交叉验证集上的 AUC、accuracy、样本数等。
* `train_index`：训练集样本的 MultiIndex[date, symbol]。
* `valid_index`：交叉验证集样本的 MultiIndex[date, symbol]。
* `feature_columns`：最终输入模型的特征列名。
* `preprocessing_summary`：预处理过程中的摘要信息，例如样本数、缺失率、PCA 解释方差等。

# 四、请提前声明并实现以下函数

## 1. get_month_end_dates

```python
def get_month_end_dates(
    trading_dates: pd.Index,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """
    从交易日序列中提取每个自然月的最后一个交易日。
    """
```

输入：

* `trading_dates`：交易日索引。
* `start_date`、`end_date`：可选的开始和结束日期。

输出：

* 月末交易日序列。

要求：

* 返回值必须升序排列。
* 对每个自然月只取最后一个交易日。
* 真实项目路径中，如果已经通过 `SelectionPipeline.load_data()` 得到 `signal_dates`，优先直接使用这些调仓信号日；本函数主要作为纯函数 fallback 或模拟数据示例使用。

## 2. align_month_end_panel

```python
def align_month_end_panel(
    factor_panel: pd.DataFrame,
    month_ends: pd.DatetimeIndex,
    factor_cols: list[str],
) -> pd.DataFrame:
    """
    对齐月末因子暴露。
    """
```

输入：

* `factor_panel`：MultiIndex[date, symbol] 的原始因子暴露。
* `month_ends`：月末交易日。
* `factor_cols`：需要使用的因子列。

输出：

```python
X_raw: pd.DataFrame
```

格式：

```text
index: MultiIndex[date, symbol]
columns: factor_cols
```

要求：

* 只保留 `month_ends` 中的日期。
* 只保留 `factor_cols`。
* 不允许使用未来日期的因子数据。
* 真实 Alpha158 数据路径中，可以参考 `load_alpha158_on_signal_dates(db_path, signal_dates, factor_names, symbols)` 直接得到已对齐的 `factor_panel`；此函数主要用于外部已传入面板后的二次筛选。

## 3. compute_forward_returns

```python
def compute_forward_returns(
    close: pd.DataFrame | None = None,
    month_ends: pd.DatetimeIndex | None = None,
    market_data: pd.DataFrame | None = None,
    signal_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    计算每个月末到下一月末的个股前瞻收益。
    """
```

输入：

* `market_data`、`signal_dates`：当前框架标准行情数据和调仓信号日；两者同时提供时优先使用。
* `close`、`month_ends`：纯 DataFrame fallback，用于外部已给定价格矩阵或模拟数据示例。

输出：

```text
index: current_month_end
columns: symbol
values: close[next_month_end] / close[current_month_end] - 1
```

要求：

* 最后一个月由于没有下一月收益，应为 NaN 或被后续删除。
* 不要使用日内未来信息，只使用月末收盘价。
* 对停牌或缺失价格导致无法计算收益的股票返回 NaN。
* 如果已有标准 `market_data` 和 `signal_dates`，优先调用 `calc_factors_returns(market_data, signal_dates)` 复用当前评估框架的调仓收益计算逻辑，再将返回的 MultiIndex Series 转为 date x symbol 矩阵。
* 只有未提供 `market_data` 或 `signal_dates` 时，才使用 `close` 和 `month_ends` 手工计算。

## 4. compute_forward_excess_returns

```python
def compute_forward_excess_returns(
    stock_forward_returns: pd.DataFrame,
    benchmark_close: pd.Series,
    month_ends: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    计算股票下一月相对基准的超额收益。
    """
```

输出：

```text
index: current_month_end
columns: symbol
values: stock_forward_return - benchmark_forward_return
```

要求：

* 基准收益同样使用 `benchmark_close[next_month_end] / benchmark_close[current_month_end] - 1`。
* 与股票收益按当前月末日期对齐。
* `benchmark_close` 使用 SPY500 ETF 的收盘价序列即可，例如 symbol 为 `SPY` 的 S&P 500 ETF 数据。

## 5. build_classification_labels

```python
def build_classification_labels(
    forward_excess_returns: pd.DataFrame,
    top_quantile: float = 0.30,
    bottom_quantile: float = 0.30,
    min_count_per_date: int = 30,
) -> pd.Series:
    """
    按月度截面构造 SVM 二分类标签。
    """
```

输出：

```python
y: pd.Series
```

格式：

```text
index: MultiIndex[date, symbol]
values: 1 或 -1
```

要求：

* 对每个日期单独做全市场等权截面排序。
* 下月超额收益位于截面前 `top_quantile` 的股票，标签为 `1`。
* 下月超额收益位于截面后 `bottom_quantile` 的股票，标签为 `-1`。
* 中间样本不进入训练集，不要输出标签。
* 若某月有效股票数少于 `min_count_per_date`，则跳过该月。
* 每只股票在分层排序中的权重相同，不使用任何分组字段。
* 注意这里的标签使用 T+1 月收益，但样本日期仍然标记为 T 月月末。

## 6. winsorize_mad_by_date

```python
def winsorize_mad_by_date(
    X: pd.DataFrame,
    n_mad: float = 5.0,
) -> pd.DataFrame:
    """
    对每个日期、每个因子做中位数 MAD 去极值。
    """
```

要求：

对每个截面、每个因子列：

```text
median = 当前日期该因子的截面中位数
mad = median(abs(x - median))
upper = median + n_mad * mad
lower = median - n_mad * mad
```

将超过上下界的值截断到边界。

边界情况：

* 若 `mad == 0` 或有效样本太少，则该因子该截面不做截断。
* 保持原始 index 和 columns 不变。

## 7. fill_missing_by_market

```python
def fill_missing_by_market(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    用同日期全市场等权统计量填充缺失值。
    """
```

要求：

* 对每个日期、每个因子单独处理。
* 某股票某因子缺失时，使用该日期全市场该因子的等权中位数填充。
* 如果全市场中位数仍不可用，则填充为 0。
* 不允许跨日期向前或向后填充，避免未来函数。

## 8. neutralize_by_size

```python
def neutralize_by_size(
    X: pd.DataFrame,
    market_cap: pd.DataFrame,
) -> pd.DataFrame:
    """
    对每个日期、每个因子做全市场等权市值中性化，返回回归残差。
    """
```

要求：

对每个日期、每个因子，做横截面 OLS：

```text
factor_value ~ log_market_cap
```

然后取残差作为新的因子暴露。

注意：

* OLS 使用全市场截面有效样本，每只股票权重相同。
* `market_cap <= 0` 的样本不能取 log，应视为缺失。
* 若有效样本数量不足，例如少于 `max(30, 自变量数量 + 5)`，则该因子该日期跳过中性化，直接返回原值或只做去均值处理。
* 不能使用未来市值。

## 9. standardize_by_date

```python
def standardize_by_date(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    对每个日期、每个因子做截面 z-score 标准化。
    """
```

要求：

```text
z = (x - mean) / std
```

边界情况：

* 若标准差为 0 或 NaN，则该因子该日期返回 0。
* 保持 index 和 columns 不变。

## 10. preprocess_features

```python
def preprocess_features(
    X_raw: pd.DataFrame,
    market_cap: pd.DataFrame,
    use_pca: bool = True,
    pca_n_components: int | None = None,
    pca_random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    完成 PCA 之前的全部横截面预处理：
    1. 中位数 MAD 去极值
    2. 全市场等权中位数填充缺失值
    3. 全市场等权市值中性化
    4. 截面标准化

    注意：PCA 不在这个函数内拟合，以便只在训练集上 fit。
    """
```

输出：

```python
X_processed, summary
```

要求：

* `X_processed` 仍然是 MultiIndex[date, symbol]。
* 不在这里拟合 PCA。
* `summary` 至少包含处理前后缺失率、样本数、因子数。

## 11. make_supervised_dataset
```python
def make_supervised_dataset(
    X_processed: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    将特征和标签按 MultiIndex[date, symbol] 对齐，得到监督学习样本。
    """
```

要求：

- 只保留同时拥有特征和标签的样本。
- 删除仍包含 NaN 或 inf 的样本。
- 返回的 X_model 和 y_model index 必须完全一致。
- y_model 只能包含 1 和 -1。

## 12. split_train_valid

```python
def split_train_valid(
    X: pd.DataFrame,
    y: pd.Series,
    valid_size: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    将样本随机划分为训练集和交叉验证集。
    """
```

要求：

* 使用 stratify，保证训练集和验证集正负样本比例接近。
* 默认 90% 训练，10% 交叉验证。
* 返回：

```python
X_train, X_valid, y_train, y_valid
```

## 13. fit_pca_on_train

```python
def fit_pca_on_train(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    use_pca: bool = True,
    n_components: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, PCA | None, list[str], dict[str, Any]]:
    """
    只在训练集上拟合 PCA，然后同时转换训练集和验证集。
    """
```

要求：

* 如果 `use_pca=False`，则原样返回 `X_train`、`X_valid`，`pca=None`。
* 如果 `use_pca=True`：

  * 只对 `X_train` 调用 `pca.fit`。
  * 对 `X_train` 和 `X_valid` 调用 `pca.transform`。
  * 如果 `n_components is None`，默认保留与原始因子数相同的维度。
  * 输出列名为 `pc_001`, `pc_002`, ...
* 返回：

```python
X_train_final, X_valid_final, pca, feature_columns, pca_summary
```

严禁在训练集和验证集合并后再 fit PCA，避免验证集信息泄漏。

## 14. evaluate_svm_classifier

```python
def evaluate_svm_classifier(
    model: SVC,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """
    评估 SVM 分类模型。
    """
```

要求：

* 使用 `model.decision_function(X)` 作为连续预测值。
* 计算：

  * `auc`
  * `accuracy`
  * `positive_count`
  * `negative_count`
  * `sample_count`
* AUC 用 `sklearn.metrics.roc_auc_score`。
* 因为 y 是 `-1` 和 `1`，计算 AUC 时应确保正类为 `1`。
* accuracy 用 `model.predict(X)` 与 `y` 比较。

## 15. grid_search_rbf_svm

```python
def grid_search_rbf_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    C_grid: list[float],
    gamma_grid: list[float],
    class_weight: str | dict | None = None,
    random_state: int = 42,
) -> tuple[SVC, dict[str, float], pd.DataFrame]:
    """
    对高斯核 SVM 做 C 和 gamma 的网格搜索。
    """
```

要求：

* 使用 `sklearn.svm.SVC(kernel="rbf")`。
* 对每组 `(C, gamma)`：

  * 在训练集上拟合模型。
  * 在交叉验证集上计算 AUC 和 accuracy。
* 选择交叉验证集 AUC 最高的参数。
* 如果 AUC 并列，选择 accuracy 更高者。
* 如果仍并列，选择 C 和 gamma 更小者，以降低过拟合风险。
* 返回：

  * 最优模型
  * 最优参数
  * 所有参数组合的验证结果 DataFrame

`cv_results` 至少包含：

```text
C
gamma
valid_auc
valid_accuracy
train_auc
train_accuracy
valid_sample_count
train_sample_count
```

## 16. train_final_model

```python
def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict[str, float],
    class_weight: str | dict | None = None,
    random_state: int = 42,
) -> SVC:
    """
    使用最优参数在给定样本上训练最终高斯核 SVM。
    """
```

要求：

* 使用 `SVC(kernel="rbf", C=best_params["C"], gamma=best_params["gamma"])`。
* 默认不启用 `probability=True`，因为后续选股排序可以直接使用 `decision_function`。
* 返回训练好的模型。

## 17. predict_svm_scores

```python
def predict_svm_scores(
    model: SVC,
    X: pd.DataFrame,
) -> pd.Series:
    """
    输出 SVM 判别函数值 f(x)，作为模型预测分数。
    """
```

输出：

```text
index: X.index
values: decision_function score
```

要求：

* 分数越大，表示模型越认为该股票更可能属于下月高收益组。
* 不要直接输出分类标签作为选股因子，应该输出连续的 `decision_function` 分数。

## 18. run_svm_training_pipeline

```python
def run_svm_training_pipeline(
    factor_panel: pd.DataFrame,
    close: pd.DataFrame | None,
    benchmark_close: pd.Series,
    market_cap: pd.DataFrame,
    factor_cols: list[str],
    market_data: pd.DataFrame | None = None,
    signal_dates: pd.DatetimeIndex | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    top_quantile: float = 0.30,
    bottom_quantile: float = 0.30,
    valid_size: float = 0.10,
    use_pca: bool = True,
    pca_n_components: int | None = None,
    C_grid: list[float] | None = None,
    gamma_grid: list[float] | None = None,
    class_weight: str | dict | None = None,
    random_state: int = 42,
) -> SVMTrainingResult:
    """
    串联完整训练流程。
    """
```

默认参数：

```python
if C_grid is None:
    C_grid = [0.01, 0.1, 1.0, 10.0, 100.0]

if gamma_grid is None:
    gamma_grid = [0.0001, 0.001, 0.01, 0.1, 1.0]
```

流程：

1. 优先使用 `signal_dates`；如果未提供，则从 `close.index` 中提取月末交易日。
2. 根据 `start_date` 和 `end_date` 过滤月末日期。
3. 调用 `align_month_end_panel` 得到月末原始因子暴露；真实 Alpha158 路径中 `factor_panel` 可由 `load_alpha158_on_signal_dates` 预先生成。
4. 调用 `compute_forward_returns` 计算个股下一月收益；如果传入 `market_data` 和 `signal_dates`，应复用 `calc_factors_returns`。
5. 调用 `compute_forward_excess_returns` 计算下一月超额收益。
6. 调用 `build_classification_labels` 构造 SVM 标签。
7. 调用 `preprocess_features` 完成去极值、全市场等权缺失填充、市值中性化、标准化。
8. 调用 `make_supervised_dataset` 对齐 X 和 y。
9. 调用 `split_train_valid` 划分训练集和交叉验证集。
10. 调用 `fit_pca_on_train`，只在训练集上拟合 PCA。
11. 调用 `grid_search_rbf_svm` 选择最优 C 和 gamma。
12. 返回 `SVMTrainingResult`。

注意：

* 默认返回的 `model` 可以是网格搜索中已经在训练集上拟合好的最优模型。
* 不要把交叉验证集再合并进训练集重训，除非额外提供参数 `refit_on_full_sample=True`。本版本先不要实现这个参数，保持简单。
* 所有步骤要保证 index 对齐清晰。

# 五、避免未来函数的要求

请在代码注释中明确说明以下约束：

1. T 月特征只能来自 T 月月末及以前。
2. T 月标签可以用 T+1 月收益构造，但只能用于训练阶段，不能用于 T 月实时预测。
3. 缺失值填充、中性化、标准化都只能在同一个 T 月全市场等权横截面内完成，不能跨日期使用未来数据。
4. PCA 只能在训练集上 `fit`，然后用于转换验证集。
5. 参数选择只能看交叉验证集 AUC，不能使用样本外测试期或回测表现参与调参。

# 六、代码质量要求

1. 每个函数都要有 docstring。
2. 对输入 index 类型做必要检查。
3. 对 MultiIndex 的层名做兼容处理：

   * 如果没有层名，则默认第一层是 date，第二层是 symbol。
   * 最好统一命名为 `date` 和 `symbol`。
4. 对异常情况给出明确错误信息，例如：

   * 没有可用月末日期。
   * 标签只有一个类别。
   * 训练样本数太少。
   * 特征全为空。
5. 不要在函数内部静默吞掉异常。
6. 保留类型标注。
7. 尽量不要写过度复杂的类，训练流程用函数和 dataclass 即可。
8. 请补充一个 `if __name__ == "__main__":` 下的最小示例，但示例可以用随机模拟数据，不需要真实行情数据。

# 七、请生成的文件

请生成一个文件：

```text
scripts/train_rbf_svm_stock_selection.py
```

文件中包含：

* imports
* `SVMTrainingResult`
* 上述所有函数
* 一个最小可运行的模拟数据示例

# 八、实现细节补充

对于 `neutralize_by_size`，可以使用 `numpy.linalg.lstsq` 或 `statsmodels`。优先使用 numpy，减少依赖。

对于每个因子的中性化：

```text
y = 当前因子暴露
X_reg = log_market_cap
residual = y - X_reg @ beta
```

返回 residual。

对于 SVM：

```python
from sklearn.svm import SVC

model = SVC(
    kernel="rbf",
    C=C,
    gamma=gamma,
    class_weight=class_weight,
    random_state=random_state,
)
```

虽然 `SVC` 的 RBF 训练本身不依赖 `random_state`，但仍保留参数，方便接口统一。

# 九、最终输出前请自检

请在完成代码后自检：

1. `run_svm_training_pipeline` 能否在模拟数据上跑通。
2. `cv_results` 是否按 `valid_auc` 从高到低排序。
3. `best_params` 是否与 `cv_results` 第一行一致。
4. `X_train`、`X_valid`、`y_train`、`y_valid` 是否 index 对齐。
5. PCA 是否只在训练集 fit。
6. `predict_svm_scores` 是否返回连续分数，而不是分类标签。
7. 是否存在明显未来函数。
