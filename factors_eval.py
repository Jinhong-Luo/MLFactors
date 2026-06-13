"""因子 IC、换手率和分层收益等基础评估指标。"""

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy import stats


warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)


@dataclass
class LayeredResult:
    """保存分层回测产生的收益和风险指标。

    Attributes
    ----------
    group_returns : pd.DataFrame
        每个调仓日期、每个因子分组的等权平均收益，索引为日期，列为组号。
    cumulative_returns : pd.DataFrame
        根据 ``group_returns`` 复利计算的累计收益。
    annual_returns : pd.Series
        各组按照实际有效期数和年化频率计算的年化收益。
    sharpe_ratios : pd.Series
        各组不扣除无风险利率的年化夏普比率。
    long_max_drawdown : float
        最高因子分组作为多头组合时的最大回撤。
    short_max_drawdown : float
        最低因子分组取反作为空头组合时的最大回撤。
    top_max_drawdown, bottom_max_drawdown : float
        与 long/short 最大回撤相同的兼容字段。
    top_excess_annual : float
        最高组相对所有组等权平均收益的年化超额收益。
    top_excess_max_drawdown : float
        最高组超额收益曲线的最大回撤。
    top_excess_calmar : float
        年化超额收益除以超额最大回撤绝对值。
    n_groups : int
        分层数量。
    """

    group_returns: pd.DataFrame
    cumulative_returns: pd.DataFrame
    annual_returns: pd.Series
    sharpe_ratios: pd.Series
    long_max_drawdown: float = 0.0
    short_max_drawdown: float = 0.0
    top_max_drawdown: float = 0.0
    bottom_max_drawdown: float = 0.0
    top_excess_annual: float = 0.0
    top_excess_max_drawdown: float = 0.0
    top_excess_calmar: float = 0.0
    n_groups: int = 5


@dataclass
class FactorEvalResult:
    """保存单个因子的完整评估结果。

    Attributes
    ----------
    summary : pd.DataFrame
        单行核心指标汇总表。
    ic_series : pd.Series
        每个日期截面的 IC 时间序列。
    turnover : pd.Series
        最高因子分组组合的单边换手率。
    forward_returns : pd.Series
        与因子值对齐使用的未来收益。
    layered : LayeredResult
        分层收益和风险指标。
    ic_decay : pd.Series
        从 1 到 ``max_lag`` 的平均 IC 衰减曲线。
    """

    summary: pd.DataFrame
    ic_series: pd.Series
    turnover: pd.Series
    forward_returns: pd.Series
    layered: LayeredResult
    ic_decay: pd.Series


def calc_ic(
    factor: pd.Series,
    returns: pd.Series,
    method: str = "rank",
) -> float:
    """计算一个日期截面上的因子值与未来收益相关系数。

    函数先按索引对齐因子值和收益，再删除任一侧为 NaN 的样本。
    有效股票少于 3 只时无法形成可靠截面，返回 ``np.nan``。

    Parameters
    ----------
    factor : pd.Series
        同一个日期截面上各股票的因子值，索引通常为 symbol。
    returns : pd.Series
        与 ``factor`` 对应的未来收益。
    method : str, default "rank"
        ``"rank"`` 使用 Spearman 秩相关；其他值使用 Pearson 线性相关。

    Returns
    -------
    float
        截面 IC，理论范围为 [-1, 1]；有效样本不足时为 NaN。
    """
    aligned = pd.DataFrame({"factor": factor, "returns": returns}).dropna()
    if len(aligned) < 3:
        return np.nan
    if method == "rank":
        return float(stats.spearmanr(aligned["factor"], aligned["returns"])[0])
    return float(stats.pearsonr(aligned["factor"], aligned["returns"])[0])


def calc_ic_series(
    factor: pd.DataFrame | pd.Series,
    returns: pd.DataFrame | pd.Series,
    method: str = "rank",
) -> pd.Series:
    """对每个日期截面计算 IC，形成按日期排列的 IC 时间序列。

    输入应使用 ``(date, symbol)`` MultiIndex。若传入 DataFrame，函数只使用
    第一列。因子和收益先按完整 MultiIndex 对齐并删除缺失值，再按第一层
    date 分组调用 :func:`calc_ic`。没有任何共同样本时返回空 Series。

    Parameters
    ----------
    factor : pd.Series or pd.DataFrame
        多期因子值，索引为 ``(date, symbol)``。
    returns : pd.Series or pd.DataFrame
        与因子对应的多期未来收益，索引结构相同。
    method : str, default "rank"
        IC 计算方式，``"rank"`` 为 Spearman，否则为 Pearson。

    Returns
    -------
    pd.Series
        名称为 ``IC``、索引为日期的升序时间序列。
    """
    if isinstance(factor, pd.DataFrame):
        factor = factor.iloc[:, 0]
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]

    combined = pd.DataFrame({"factor": factor, "returns": returns}).dropna()
    if combined.empty:
        return pd.Series(dtype=float, name="IC")

    def cross_sectional_ic(cross_section: pd.DataFrame) -> float:
        """计算 groupby 传入的单日截面 IC。

        ``cross_section`` 包含已经对齐并删除缺失值的 factor 和 returns
        两列，返回当前日期的相关系数。
        """
        return calc_ic(cross_section["factor"], cross_section["returns"], method)

    result = combined.groupby(level=0).apply(cross_sectional_ic)
    return pd.Series(result, name="IC").sort_index()


def calc_icir(
    ic_series: pd.Series,
    period: int = 1,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> float:
    """计算 IC 信息比率，并修正多期重叠收益造成的虚高。

    基础 ICIR 为 ``mean(IC) / std(IC)``。当未来收益覆盖多个周期时，结果
    再除以 ``sqrt(period)``。若要求年化，则乘以
    ``sqrt(periods_per_year)``。少于两个有效 IC 或标准差为零时返回 NaN。

    Parameters
    ----------
    ic_series : pd.Series
        按时间排列的 IC 序列。
    period : int, default 1
        每个未来收益覆盖的周期数，用于重叠收益修正。
    annualize : bool, default False
        是否对修正后的 ICIR 进行年化。
    periods_per_year : int, default 252
        一年的基础周期数，日频通常使用 252。

    Returns
    -------
    float
        修正后或年化后的 ICIR。
    """
    values = ic_series.dropna()
    if len(values) < 2 or values.std() == 0:
        return np.nan
    result = values.mean() / values.std() / np.sqrt(period)
    if annualize:
        result *= np.sqrt(periods_per_year)
    return float(result)


def calc_t_stat(ic_series: pd.Series) -> tuple[float, float]:
    """检验 IC 均值是否显著偏离零。

    使用 SciPy 单样本 t 检验，以零为原假设均值。计算前删除 NaN；有效
    样本少于 2 个时返回 ``(np.nan, np.nan)``。

    Parameters
    ----------
    ic_series : pd.Series
        IC 时间序列。

    Returns
    -------
    tuple[float, float]
        ``(t_stat, p_value)``，分别为 t 统计量和双侧 p-value。
    """
    values = ic_series.dropna()
    if len(values) < 2:
        return np.nan, np.nan
    t_stat, p_value = stats.ttest_1samp(values, 0)
    return float(t_stat), float(p_value)


def calc_ic_decay(
    factor: pd.DataFrame | pd.Series,
    returns_provider,
    max_lag: int = 20,
    method: str = "rank",
) -> pd.Series:
    """计算因子对不同未来滞后收益的平均 IC，观察预测能力衰减。

    ``lag=1`` 使用原始收益，``lag=2`` 将宽表收益向前移动一个日期，以此
    类推。收益既可以直接传入，也可以由函数或按 lag 索引的对象动态提供。
    DataFrame 输入只使用第一列。

    Parameters
    ----------
    factor : pd.Series or pd.DataFrame
        ``(date, symbol)`` MultiIndex 因子值。
    returns_provider : pd.Series, pd.DataFrame, callable or mapping
        未来收益数据。可调用对象接收 lag 并返回收益；映射使用 lag 取值。
    max_lag : int, default 20
        需要计算的最大滞后期，结果包含 1 到 ``max_lag``。
    method : str, default "rank"
        每个截面使用的 IC 计算方法。

    Returns
    -------
    pd.Series
        索引为 lag、值为对应 IC 时间序列均值，名称为 ``IC_decay``。
    """
    if isinstance(factor, pd.DataFrame):
        factor = factor.iloc[:, 0]

    result = {}
    if isinstance(returns_provider, (pd.DataFrame, pd.Series)):
        returns = returns_provider.iloc[:, 0] if isinstance(returns_provider, pd.DataFrame) else returns_provider
        returns = returns.unstack()
        for lag in range(1, max_lag + 1):
            shifted_returns = returns.shift(-(lag - 1)).stack()
            result[lag] = calc_ic_series(factor, shifted_returns, method).mean()
    else:
        for lag in range(1, max_lag + 1):
            returns = returns_provider(lag) if callable(returns_provider) else returns_provider[lag]
            result[lag] = calc_ic_series(factor, returns, method).mean()
    return pd.Series(result, name="IC_decay")


def calc_turnover(
    factor: pd.DataFrame | pd.Series,
    quantiles: int = 5,
) -> pd.Series:
    """计算最高因子分位数组合相邻调仓期之间的单边换手率。

    每个日期先对有效因子值使用 ``rank(method="first")``，再通过 qcut 分成
    ``quantiles`` 组。最高组内股票等权，其余股票权重为零。单边换手率为
    ``sum(abs(weight_t - weight_t-1)) / 2``。截面股票数不足分组数时，该期
    所有权重设为零。

    Parameters
    ----------
    factor : pd.Series or pd.DataFrame
        ``(date, symbol)`` MultiIndex 因子值；DataFrame 只使用第一列。
    quantiles : int, default 5
        截面分组数量，5 表示持有最高 20% 的股票。

    Returns
    -------
    pd.Series
        索引为日期、名称为 ``turnover`` 的单边换手率序列。
    """
    if isinstance(factor, pd.DataFrame):
        factor = factor.iloc[:, 0]

    def top_group_weights(cross_section: pd.Series) -> pd.Series:
        """为单日最高因子分组生成等权权重。

        输入为一个日期上所有股票的因子值，返回索引完全相同的权重
        Series。最高分位组等权，其余股票以及缺失样本权重为零。
        """
        values = cross_section.dropna()
        if len(values) < quantiles:
            return pd.Series(0.0, index=cross_section.index)
        groups = pd.qcut(values.rank(method="first"), quantiles, labels=False)
        selected = values[groups == groups.max()]
        weights = pd.Series(1.0 / len(selected), index=selected.index)
        return weights.reindex(cross_section.index).fillna(0.0)

    weights = factor.unstack().apply(top_group_weights, axis=1)
    turnover = weights.fillna(0.0).diff().abs().sum(axis=1) / 2.0
    turnover.name = "turnover"
    return turnover.dropna()


def calc_forward_returns(
    market_data: pd.DataFrame,
    period: int,
    price_col: str = "close",
) -> pd.Series:
    """根据价格面板计算指定周期的未来持有收益。

    计算口径为在因子日后的第一个交易日价格买入，并在再向后 ``period``
    个交易日的价格卖出：``price[t+1+period] / price[t+1] - 1``。这种口径
    避免直接使用因子形成日收盘价成交。结果尾部因缺少未来价格会自然缺失。

    Parameters
    ----------
    market_data : pd.DataFrame
        索引为 ``(date, symbol)`` 的行情数据。
    period : int
        未来持有交易日数量，必须为正整数。
    price_col : str, default "close"
        用于计算收益的价格列。

    Returns
    -------
    pd.Series
        ``(date, symbol)`` MultiIndex 收益 Series，名称为
        ``fwd_ret_<period>``。

    Raises
    ------
    ValueError
        ``period`` 不是正整数时抛出。
    """
    if period <= 0:
        raise ValueError("period 必须为正整数")
    price = market_data[price_col].unstack()
    return (
        price.shift(-(1 + period)) / price.shift(-1) - 1
    ).stack().rename(f"fwd_ret_{period}")


def calc_max_drawdown(returns: pd.Series) -> float:
    """根据周期收益序列计算复利净值的最大回撤。

    先删除缺失收益，再计算 ``cumprod(1 + returns)`` 得到财富曲线。回撤为
    当前财富相对历史峰值的跌幅，返回其中最小值，因此结果通常小于或等于
    零。空收益序列返回 0。

    Parameters
    ----------
    returns : pd.Series
        按时间排序的周期收益率。

    Returns
    -------
    float
        最大回撤，例如 ``-0.2`` 表示从峰值下跌 20%。
    """
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    wealth = (1 + returns).cumprod()
    drawdown = (wealth - wealth.cummax()) / wealth.cummax()
    return float(drawdown.min())


def layered_backtest(
    factor: pd.DataFrame | pd.Series,
    returns: pd.DataFrame | pd.Series,
    n_groups: int = 5,
    annual_trading_days: int = 252,
    period: int = 1,
) -> LayeredResult:
    """按因子值进行截面分层，并汇总各层收益和风险指标。

    每个日期先对因子值排名，以 ``method="first"`` 打破相同因子值的并列，
    然后用 qcut 划分为 1 到 ``n_groups``。组号越大表示因子值越高，各组
    收益为组内股票的等权平均收益。函数进一步计算累计收益、年化收益、
    年化夏普、最高组多头最大回撤、最低组取反后的空头最大回撤，以及最高
    组相对全部组等权基准的超额收益、回撤和 Calmar 比率。

    Parameters
    ----------
    factor : pd.Series or pd.DataFrame
        ``(date, symbol)`` MultiIndex 因子值；DataFrame 只使用第一列。
    returns : pd.Series or pd.DataFrame
        与因子索引对齐的未来收益；DataFrame 只使用第一列。
    n_groups : int, default 5
        每个日期截面的分组数量。
    annual_trading_days : int, default 252
        基础数据每年的交易周期数量。
    period : int, default 1
        每条收益覆盖的周期数。年化频率为
        ``annual_trading_days / period``，收益本身不会被 period 除算。

    Returns
    -------
    LayeredResult
        包含逐期分组收益、累计收益及各类汇总指标。

    Raises
    ------
    ValueError
        ``period <= 0``，或输入未使用至少两层 MultiIndex 时抛出。

    Notes
    -----
    调用方应确保输入收益与实际调仓频率一致。若把每天都存在的重叠多日
    forward return 直接复利，会高估可交易组合的累计表现。
    """
    if period <= 0:
        raise ValueError("period 必须为正整数")
    if isinstance(factor, pd.DataFrame):
        factor = factor.iloc[:, 0]
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]

    combined = pd.DataFrame({"factor": factor, "returns": returns}).dropna()
    if not isinstance(combined.index, pd.MultiIndex) or combined.index.nlevels < 2:
        raise ValueError("factor 和 returns 必须使用 MultiIndex: (date, symbol)")

    records = []
    for current_date, cross_section in combined.groupby(level=0):
        cross_section = cross_section.droplevel(0)
        if len(cross_section) < n_groups:
            continue
        groups = pd.qcut(
            cross_section["factor"].rank(method="first"),
            n_groups,
            labels=False,
            duplicates="drop",
        ) + 1
        for group in range(1, n_groups + 1):
            group_returns = cross_section.loc[groups == group, "returns"]
            if not group_returns.empty:
                records.append((current_date, group, group_returns.mean()))

    if not records:
        return LayeredResult(
            group_returns=pd.DataFrame(),
            cumulative_returns=pd.DataFrame(),
            annual_returns=pd.Series(dtype=float),
            sharpe_ratios=pd.Series(dtype=float),
            n_groups=n_groups,
        )

    group_returns = pd.DataFrame(
        records, columns=["date", "group", "returns"]
    ).pivot(index="date", columns="group", values="returns").sort_index()
    periods_per_year = annual_trading_days / period
    cumulative_returns = (1 + group_returns).cumprod() - 1
    total_returns = (1 + group_returns).prod(skipna=True)
    valid_periods = group_returns.notna().sum().clip(lower=1)
    annual_returns = total_returns.where(total_returns > 0) ** (
        periods_per_year / valid_periods
    ) - 1
    annual_returns.name = "annual_return"

    group_std = group_returns.std(ddof=1)
    sharpe_ratios = (
        group_returns.mean().divide(group_std.where(group_std > 0))
        * np.sqrt(periods_per_year)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sharpe_ratios.name = "sharpe_ratio"

    top_group = int(group_returns.columns.max())
    bottom_group = int(group_returns.columns.min())
    long_max_drawdown = calc_max_drawdown(group_returns[top_group])
    short_max_drawdown = calc_max_drawdown(-group_returns[bottom_group])

    benchmark_returns = group_returns.mean(axis=1)
    excess_returns = (group_returns[top_group] - benchmark_returns).dropna()
    if excess_returns.empty:
        excess_annual = 0.0
        excess_max_drawdown = 0.0
    else:
        excess_total = (1 + excess_returns).prod(skipna=True)
        excess_annual = (
            float(excess_total ** (periods_per_year / len(excess_returns)) - 1)
            if excess_total > 0
            else np.nan
        )
        excess_max_drawdown = calc_max_drawdown(excess_returns)
    excess_calmar = (
        float(excess_annual / abs(excess_max_drawdown))
        if np.isfinite(excess_annual) and excess_max_drawdown < 0
        else 0.0
    )

    return LayeredResult(
        group_returns=group_returns,
        cumulative_returns=cumulative_returns,
        annual_returns=annual_returns,
        sharpe_ratios=sharpe_ratios,
        long_max_drawdown=long_max_drawdown,
        short_max_drawdown=short_max_drawdown,
        top_max_drawdown=long_max_drawdown,
        bottom_max_drawdown=short_max_drawdown,
        top_excess_annual=excess_annual,
        top_excess_max_drawdown=excess_max_drawdown,
        top_excess_calmar=excess_calmar,
        n_groups=n_groups,
    )

def eval(
    factor_values: pd.DataFrame | pd.Series,
    market_data: pd.DataFrame,
    forward_period: int = 1,
    n_groups: int = 5,
    ic_method: str = "rank",
    max_lag: int = 20,
) -> FactorEvalResult:
    """汇总单个因子的 IC、换手率和分层回测指标。

    函数根据 ``market_data`` 计算指定周期的未来收益，并按照
    ``forward_period`` 对交易日期进行等间隔采样，再将因子值与收益对齐，
    汇总 IC、换手率、分层收益和风险指标。日期采样确保多日未来收益不会
    因每日调仓而发生持有期重叠。DataFrame 因子输入应为 ``date × symbol``
    宽表；Series 输入应使用 ``(date, symbol)`` MultiIndex。

    Parameters
    ----------
    factor_values : pd.DataFrame or pd.Series
        因子计算结果。DataFrame 的索引为 date、列为 symbol；Series 的
        索引为 ``(date, symbol)``。
    market_data : pd.DataFrame
        ``(date, symbol)`` MultiIndex 行情数据，至少包含 ``close`` 列。
    forward_period : int, default 1
        未来收益持有周期和调仓间隔。函数每隔该数量的交易日选择一次
        因子截面，同时用于 ICIR 和分层收益年化频率修正。
    n_groups : int, default 5
        每个日期截面的分组数量。
    ic_method : str, default "rank"
        ``"rank"`` 计算 Spearman RankIC；其他值计算 Pearson IC。
    max_lag : int, default 20
        IC 衰减曲线计算的最大滞后期。

    Returns
    -------
    FactorEvalResult
        完整评估结果，包含汇总表、IC 序列、换手率、未来收益、分层结果
        和 IC 衰减曲线，可直接交给绘图模块使用。

    Raises
    ------
    ValueError
        ``forward_period`` 或 ``n_groups`` 不是正整数时抛出。
    """
    if forward_period <= 0:
        raise ValueError("forward_period 必须为正整数")
    if n_groups <= 0:
        raise ValueError("n_groups 必须为正整数")
    if max_lag <= 0:
        raise ValueError("max_lag 必须为正整数")

    if isinstance(factor_values, pd.DataFrame):
        factor = factor_values.stack().rename("factor")
    else:
        factor = factor_values.rename("factor")

    trading_dates = pd.DatetimeIndex(
        market_data.index.get_level_values("date").unique()
    ).sort_values()
    sampled_dates = trading_dates[::forward_period]
    factor = factor[
        factor.index.get_level_values("date").isin(sampled_dates)
    ]

    forward_returns = calc_forward_returns(market_data, forward_period)
    forward_returns = forward_returns[
        forward_returns.index.get_level_values("date").isin(sampled_dates)
    ]
    ic_series = calc_ic_series(factor, forward_returns, method=ic_method)
    turnover = calc_turnover(factor, quantiles=n_groups)
    layered = layered_backtest(
        factor,
        forward_returns,
        n_groups=n_groups,
        period=forward_period,
    )
    ic_decay = calc_ic_decay(
        factor,
        forward_returns,
        max_lag=max_lag,
        method=ic_method,
    )

    t_stat, p_value = calc_t_stat(ic_series)
    long_sharpe = 0.0
    if not layered.sharpe_ratios.empty:
        long_sharpe = float(layered.sharpe_ratios.loc[layered.sharpe_ratios.index.max()])

    summary = pd.DataFrame([{
        "period": forward_period,
        "IC_mean": round(float(ic_series.mean()), 4),
        "IC_std": round(float(ic_series.std()), 4),
        "ICIR": round(
            calc_icir(ic_series, period=forward_period, annualize=True),
            4,
        ),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "IC>0_ratio": round(float((ic_series > 0).mean()), 4),
        "turnover": round(float(turnover.mean()), 4),
        "long_sharpe": round(long_sharpe, 4),
        "long_max_drawdown": round(layered.long_max_drawdown, 4),
        "short_max_drawdown": round(layered.short_max_drawdown, 4),
        "top_excess_annual": round(layered.top_excess_annual, 4),
        "top_excess_max_dd": round(layered.top_excess_max_drawdown, 4),
        "top_excess_calmar": round(layered.top_excess_calmar, 4),
    }])
    return FactorEvalResult(
        summary=summary.set_index("period"),
        ic_series=ic_series,
        turnover=turnover,
        forward_returns=forward_returns,
        layered=layered,
        ic_decay=ic_decay,
    )
