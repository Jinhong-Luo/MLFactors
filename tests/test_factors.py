"""factors 层单元测试 — BaseFactor, FactorRegistry, 内置因子。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.schema import Col, FundamentalCol
from factors.base import BaseFactor
from factors.library.selection.capm_beta import CapmBeta
from factors.registry import FactorRegistry, register_factor


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def make_market_df(n_dates: int = 30, n_symbols: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    symbols = [f"S{i:03d}" for i in range(n_symbols)]
    rng = np.random.default_rng(42)
    records = []
    for d in dates:
        for s in symbols:
            price = float(rng.uniform(9, 11))
            records.append({
                Col.DATE: d,
                Col.SYMBOL: s,
                Col.OPEN: price,
                Col.HIGH: price * 1.02,
                Col.LOW: price * 0.98,
                Col.CLOSE: price,
                Col.VOLUME: float(rng.integers(1000, 5000)),
            })
    df = pd.DataFrame(records)
    return df.set_index([Col.DATE, Col.SYMBOL]).sort_index()


def make_fundamental_df(n_dates: int = 30, n_symbols: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    symbols = [f"S{i:03d}" for i in range(n_symbols)]
    records = []
    for d in dates:
        for i, s in enumerate(symbols):
            records.append({
                FundamentalCol.DATE: d,
                FundamentalCol.SYMBOL: s,
                FundamentalCol.PB: 0.8 + i * 0.2,
                "market_cap": 1_000_000_000.0 + i * 100_000_000.0,
            })
    df = pd.DataFrame(records)
    return df.set_index([FundamentalCol.DATE, FundamentalCol.SYMBOL]).sort_index()


# ── BaseFactor 接口测试 ───────────────────────────────────────────────────────

class TestBaseFactor:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseFactor()

    def test_subclass_requires_generate_signals(self):
        class Incomplete(BaseFactor):
            name = "incomplete"

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_ok(self):
        class SimpleFactor(BaseFactor):
            name = "simple"

            def generate_signals(self, market_data, fundamental_data=None):
                return market_data[Col.CLOSE].unstack(Col.SYMBOL)

        f = SimpleFactor()
        assert f.name == "simple"

    def test_repr(self):
        class SimpleFactor(BaseFactor):
            name = "simple"
            category = "test"

            def generate_signals(self, market_data, fundamental_data=None):
                return market_data[Col.CLOSE].unstack(Col.SYMBOL)

        assert "simple" in repr(SimpleFactor())


# ── FactorRegistry 测试 ───────────────────────────────────────────────────────

class TestFactorRegistry:
    def setup_method(self):
        # 保存并隔离注册表状态，测试后还原
        self._backup = dict(FactorRegistry._registry)
        self._loaded_backup = FactorRegistry._loaded
        FactorRegistry._registry = {}
        FactorRegistry._loaded = True  # 阻止自动发现以便隔离测试

    def teardown_method(self):
        FactorRegistry._registry = self._backup
        FactorRegistry._loaded = self._loaded_backup

    def test_register_and_get(self):
        @register_factor
        class Dummy(BaseFactor):
            name = "dummy_test"

            def generate_signals(self, market_data, fundamental_data=None):
                return market_data[Col.CLOSE].unstack(Col.SYMBOL)

        assert FactorRegistry.get("dummy_test") is Dummy

    def test_list_sorted(self):
        @register_factor
        class B(BaseFactor):
            name = "b_factor"

            def generate_signals(self, market_data, fundamental_data=None):
                return market_data[Col.CLOSE].unstack(Col.SYMBOL)

        @register_factor
        class A(BaseFactor):
            name = "a_factor"

            def generate_signals(self, market_data, fundamental_data=None):
                return market_data[Col.CLOSE].unstack(Col.SYMBOL)

        names = FactorRegistry.list()
        assert names == sorted(names)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            FactorRegistry.get("nonexistent_xyz")

    def test_register_empty_name_raises(self):
        with pytest.raises(ValueError):
            @register_factor
            class NoName(BaseFactor):
                name = ""

                def generate_signals(self, market_data, fundamental_data=None):
                    return market_data[Col.CLOSE].unstack(Col.SYMBOL)

    def test_generate_all_returns_dataframe(self):
        @register_factor
        class F1(BaseFactor):
            name = "f1_test"

            def generate_signals(self, market_data, fundamental_data=None):
                return market_data[Col.CLOSE].unstack(Col.SYMBOL)

        @register_factor
        class F2(BaseFactor):
            name = "f2_test"

            def generate_signals(self, market_data, fundamental_data=None):
                return (market_data[Col.CLOSE] * 2).unstack(Col.SYMBOL)

        mkt = make_market_df()
        result = FactorRegistry.generate_all(mkt, factor_names=["f1_test", "f2_test"])
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"f1_test", "f2_test"}

    def test_list_detail_has_fields(self):
        @register_factor
        class Detail(BaseFactor):
            name = "detail_test"
            description = "test desc"
            category = "test_cat"

            def generate_signals(self, market_data, fundamental_data=None):
                return market_data[Col.CLOSE].unstack(Col.SYMBOL)

        details = FactorRegistry.list_detail()
        entry = next(d for d in details if d["name"] == "detail_test")
        assert entry["description"] == "test desc"
        assert entry["category"] == "test_cat"


# ── 内置因子测试 ──────────────────────────────────────────────────────────────

class TestBuiltinFactors:
    """验证内置因子可以正确自动注册并计算。"""

    def setup_method(self):
        FactorRegistry.reset()

    def teardown_method(self):
        FactorRegistry.reset()

    def _mkt(self):
        return make_market_df(n_dates=40, n_symbols=5)

    def test_builtin_factors_auto_registered(self):
        names = FactorRegistry.list()
        assert "momentum_5" in names
        assert "volatility_20" in names
        assert "highlow_spread_20" in names
        assert "vff3" in names

    def test_momentum5_output_shape(self):
        mkt = self._mkt()
        cls = FactorRegistry.get("momentum_5")
        result = cls().generate_signals(mkt)
        assert isinstance(result, pd.DataFrame)
        # 因 pct_change(5) 前5行为 NaN，应有非空值
        assert result.stack().dropna().__len__() > 0

    def test_momentum5_output_index(self):
        mkt = self._mkt()
        cls = FactorRegistry.get("momentum_5")
        result = cls().generate_signals(mkt)
        assert result.index.name == Col.DATE
        assert result.columns.name == Col.SYMBOL

    def test_volatility20_nonnegative(self):
        mkt = self._mkt()
        cls = FactorRegistry.get("volatility_20")
        result = cls().generate_signals(mkt).stack().dropna()
        assert (result >= 0).all()

    def test_highlow_spread_nonnegative(self):
        mkt = self._mkt()
        cls = FactorRegistry.get("highlow_spread_20")
        result = cls().generate_signals(mkt).stack().dropna()
        assert (result >= 0).all()

    def test_vff3_nonnegative(self):
        mkt = self._mkt()
        fundamental = make_fundamental_df(n_dates=40, n_symbols=5)
        cls = FactorRegistry.get("vff3")
        result = cls().generate_signals(mkt, fundamental).stack().dropna()
        assert result.__len__() > 0
        assert (result >= 0).all()


class TestCapmBeta:
    def _data_from_returns(
        self,
        returns: pd.DataFrame,
        anchor_cap: float = 1e12,
    ) -> dict[str, pd.DataFrame]:
        close = 100.0 * (1.0 + returns.fillna(0.0)).cumprod()
        close.index.name = Col.DATE
        close.columns.name = Col.SYMBOL
        market = close.stack().rename(Col.CLOSE).to_frame()
        market.index.names = [Col.DATE, Col.SYMBOL]

        market_cap = pd.DataFrame(1.0, index=close.index, columns=close.columns)
        if "MARKET" in market_cap.columns:
            market_cap["MARKET"] = anchor_cap
        market_cap.index.name = Col.DATE
        market_cap.columns.name = Col.SYMBOL
        fundamental = market_cap.stack().rename("market_cap").to_frame()
        fundamental.index.names = [Col.DATE, Col.SYMBOL]
        return {"market": market, "fundamental": fundamental}

    def test_beta_matches_known_slopes(self):
        dates = pd.date_range("2023-01-02", periods=320, freq="B")
        rng = np.random.default_rng(7)
        market_ret = pd.Series(rng.normal(0.0003, 0.01, len(dates)), index=dates)
        noise = rng.normal(0.0, 0.0005, (len(dates), 2))
        stock_ret = pd.DataFrame(
            {
                "BETA2": 2.0 * market_ret.to_numpy() + noise[:, 0],
                "BETA05": 0.5 * market_ret.to_numpy() + noise[:, 1],
                "MARKET": market_ret.to_numpy(),
            },
            index=dates,
        )
        data = self._data_from_returns(stock_ret)

        result = CapmBeta(lookback=252, min_obs=120, clip=(-10, 10), rebalance="daily").generate_signals(
            data
        )

        assert result.iloc[-1]["BETA2"] == pytest.approx(2.0, abs=0.05)
        assert result.iloc[-1]["BETA05"] == pytest.approx(0.5, abs=0.05)

    def test_zero_market_variance_returns_nan(self):
        dates = pd.date_range("2024-01-02", periods=40, freq="B")
        stock_ret = pd.DataFrame(
            {
                "AAA": np.linspace(-0.01, 0.01, len(dates)),
                "MARKET": 0.0,
            },
            index=dates,
        )
        data = self._data_from_returns(stock_ret)
        fundamental = data["fundamental"].copy()
        idx = fundamental.index.get_level_values(Col.SYMBOL) == "AAA"
        fundamental.loc[idx, "market_cap"] = 0.0
        data["fundamental"] = fundamental

        result = CapmBeta(lookback=20, min_obs=10, rebalance="daily").generate_signals(
            data
        )

        assert result["AAA"].dropna().empty

    def test_insufficient_observations_return_nan(self):
        dates = pd.date_range("2024-01-02", periods=20, freq="B")
        base_ret = pd.Series(np.linspace(-0.01, 0.01, len(dates)), index=dates)
        stock_ret = pd.DataFrame({"AAA": 1.5 * base_ret}, index=dates)
        data = self._data_from_returns(stock_ret)

        result = CapmBeta(lookback=20, min_obs=30, rebalance="daily").generate_signals(
            data
        )

        assert result["AAA"].dropna().empty

    def test_monthly_returns_last_trading_day_per_month(self):
        dates = pd.date_range("2024-01-02", periods=65, freq="B")
        base_ret = pd.Series(np.linspace(-0.01, 0.01, len(dates)), index=dates)
        stock_ret = pd.DataFrame({"AAA": 1.2 * base_ret}, index=dates)
        data = self._data_from_returns(stock_ret)

        result = CapmBeta(lookback=10, min_obs=5, rebalance="monthly").generate_signals(
            data
        )

        expected_dates = dates.to_series().groupby(dates.to_period("M")).tail(1).to_list()
        assert result.index.to_list() == expected_dates
