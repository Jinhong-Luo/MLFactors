"""本地文件数据加载器 — 支持 CSV / Parquet / SQLite / A 股标准数据目录。"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from data.base import DataLoader
from data.schema import Col, FundamentalCol

# ====================================================================
#  AStockLocalLoader — 从标准本地数据目录加载 A 股数据
# ====================================================================

class AStockLocalLoader(DataLoader):
    """从本地标准数据目录加载 A 股行情和估值数据。

    目录结构::

        <data_root>/
        ├── meta_data.duckdb             # 元数据（股票列表、交易日历）
        │   ├── securities               # 股票列表（含上市/退市日期）
        │   └── trade_calendar           # 交易日历
        ├── fundamentals.duckdb          # 基本面数据库
        │   ├── daily_valuation          # 每日估值（PE/PB/PS 等）
        │   └── financial_reports        # 财报数据（接口预留）
        └── market_data/
            └── market=A_stock/
                └── year=YYYY/
                    └── data.parquet     # 日线行情（OHLCV）

    股票过滤
    --------
    若 ``meta_data.duckdb`` 存在，加载时自动过滤无效标的：

    - 上市日期晚于 ``end`` 的股票（请求区间内尚未上市）
    - 退市日期早于 ``start`` 的股票（请求区间前已退市）

    若 ``meta_data.duckdb`` 不存在，跳过过滤，返回原始数据。

    Parameters
    ----------
    data_root : 本地数据根目录。
    """

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self._market_dir = self.data_root / "market_data"
        self._fund_db = self.data_root / "fundamentals.duckdb"
        self._meta_db = self.data_root / "meta_data.duckdb"

    # ------------------------------------------------------------------ #
    #  内部：有效股票过滤
    # ------------------------------------------------------------------ #

    def _load_valid_symbols(
        self,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> set[str] | None:
        """从 meta_data.duckdb 读取在给定日期区间内活跃的股票代码集合。

        过滤规则：
        - ``list_date <= end``：截止日前已上市
        - ``delist_date IS NULL OR delist_date >= start``：未在开始日前退市

        Returns
        -------
        活跃 symbol 集合；若 meta_data.duckdb 不存在则返回 ``None``（跳过过滤）。
        """
        if not self._meta_db.exists():
            logger.debug("meta_data.duckdb 不存在，跳过股票有效性过滤")
            return None

        conditions = ["market = 'A_stock'", "asset_type = 'stock'"]
        params: list = []
        if end is not None:
            conditions.append("(list_date IS NULL OR list_date <= ?)")
            params.append(str(pd.Timestamp(end).date()))
        if start is not None:
            conditions.append("(delist_date IS NULL OR delist_date >= ?)")
            params.append(str(pd.Timestamp(start).date()))

        query = f"SELECT symbol FROM securities WHERE {' AND '.join(conditions)}"
        try:
            with duckdb.connect(str(self._meta_db), read_only=True) as conn:
                rows = conn.execute(query, params).fetchall()
            valid = {r[0] for r in rows}
            logger.debug("meta_data: 区间内有效股票 {} 只", len(valid))
            return valid
        except Exception as exc:
            logger.warning("读取 meta_data.duckdb 失败，跳过过滤: {}", exc)
            return None

    # ------------------------------------------------------------------ #
    #  行情：Hive 分区 Parquet
    # ------------------------------------------------------------------ #

    def load_market_data(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """从 Hive 分区 Parquet 加载日线行情（OHLCV）。

        仅读取与日期区间重叠的年份分区，减少不必要的 I/O。

        Returns
        -------
        DataFrame，以 (date, symbol) 为 MultiIndex，包含
        open / high / low / close / volume / amount / adjust_flag 列。
        """
        partition_root = self._market_dir / "market=A_stock"
        if not partition_root.exists():
            logger.warning("market_data 目录不存在: {}", partition_root)
            return pd.DataFrame()

        start_year = pd.Timestamp(start).year if start is not None else None
        end_year   = pd.Timestamp(end).year   if end   is not None else None

        parquet_files = sorted(partition_root.glob("year=*/data.parquet"))
        if not parquet_files:
            logger.warning("未找到任何 Parquet 分区文件")
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for pq_file in parquet_files:
            try:
                year = int(pq_file.parent.name.split("=")[1])
            except (IndexError, ValueError):
                continue
            if start_year is not None and year < start_year:
                continue
            if end_year is not None and year > end_year:
                continue
            try:
                frames.append(pq.read_table(pq_file).to_pandas())
            except Exception as exc:
                logger.warning("读取 {} 失败，跳过: {}", pq_file, exc)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)

        # trade_date → Col.DATE ("date")
        df = df.rename(columns={"trade_date": Col.DATE})
        df[Col.DATE]   = pd.to_datetime(df[Col.DATE])
        df[Col.SYMBOL] = df[Col.SYMBOL].astype(str)

        # 与 meta_data 中有效股票取交集（过滤已退市或不存在的标的）
        valid = self._load_valid_symbols(start, end)
        if valid is not None:
            effective_symbols = list(valid if symbols is None else set(symbols) & valid)
        else:
            effective_symbols = symbols

        df = self._filter(df, effective_symbols, start, end)
        df = self._set_index(df)

        missing = set(Col.market_required()) - set(df.columns) - set(df.index.names)
        if missing:
            logger.warning("行情数据缺少列: {}", missing)

        return df

    # ------------------------------------------------------------------ #
    #  估值：fundamentals.duckdb → daily_valuation
    # ------------------------------------------------------------------ #

    def load_fundamental_data(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """从 ``fundamentals.duckdb`` 加载每日估值数据（PE/PB/PS）。

        财报数据请使用 :meth:`load_financial_reports`。

        Returns
        -------
        DataFrame，以 (date, symbol) 为 MultiIndex，包含
        pe / pe_ttm / pb / ps / ps_ttm / total_mv / circ_mv 列。
        """
        if not self._fund_db.exists():
            logger.warning("fundamentals.duckdb 不存在: {}", self._fund_db)
            return pd.DataFrame()

        conditions: list[str] = []
        params: list = []

        if symbols:
            placeholders = ",".join("?" * len(symbols))
            conditions.append(f"symbol IN ({placeholders})")
            params.extend(symbols)
        if start is not None:
            conditions.append("trade_date >= ?")
            params.append(str(pd.Timestamp(start).date()))
        if end is not None:
            conditions.append("trade_date <= ?")
            params.append(str(pd.Timestamp(end).date()))

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = (
            "SELECT symbol, trade_date, pe, pe_ttm, pb, ps, ps_ttm, total_mv, circ_mv "
            f"FROM daily_valuation{where} ORDER BY trade_date, symbol"
        )

        # 与 meta_data 有效股票取交集，追加到 SQL 过滤条件中
        valid = self._load_valid_symbols(start, end)
        if valid is not None:
            effective_symbols = list(valid if symbols is None else set(symbols) & valid)
            if effective_symbols != symbols:  # 有过滤动作时重建 WHERE
                conditions = []
                params = []
                placeholders = ",".join("?" * len(effective_symbols))
                conditions.append(f"symbol IN ({placeholders})")
                params.extend(effective_symbols)
                if start is not None:
                    conditions.append("trade_date >= ?")
                    params.append(str(pd.Timestamp(start).date()))
                if end is not None:
                    conditions.append("trade_date <= ?")
                    params.append(str(pd.Timestamp(end).date()))
                where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
                query = (
                    "SELECT symbol, trade_date, pe, pe_ttm, pb, ps, ps_ttm, total_mv, circ_mv "
                    f"FROM daily_valuation{where} ORDER BY trade_date, symbol"
                )

        try:
            with duckdb.connect(str(self._fund_db), read_only=True) as conn:
                df = conn.execute(query, params).df()
        except Exception as exc:
            logger.error("加载 daily_valuation 失败: {}", exc)
            return pd.DataFrame()

        df = df.rename(columns={
            "trade_date": FundamentalCol.DATE,
            "pe":         FundamentalCol.PE,
            "pe_ttm":     FundamentalCol.PE_TTM,
            "pb":         FundamentalCol.PB,
            "ps":         FundamentalCol.PS,
            "ps_ttm":     FundamentalCol.PS_TTM,
        })
        df[FundamentalCol.DATE]   = pd.to_datetime(df[FundamentalCol.DATE])
        df[FundamentalCol.SYMBOL] = df[FundamentalCol.SYMBOL].astype(str)

        df = self._set_index(df)
        return df

    # ------------------------------------------------------------------ #
    #  财报：接口预留
    # ------------------------------------------------------------------ #

    def load_financial_reports(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """加载财报数据（接口预留，暂未实现）。

        数据存储于 ``fundamentals.duckdb`` 的 ``financial_reports`` 表，
        字段包括：report_date, total_revenue, revenue, net_profit,
        total_assets, total_liabilities, equity, eps, bps, roe,
        gross_margin, net_margin, debt_ratio 等。

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "财报加载接口尚未实现。数据存储于 fundamentals.duckdb → "
            "financial_reports 表，可直接用 duckdb.connect() 查询。"
        )


# ====================================================================
#  USStockLocalLoader — 从本地 SQLite（us.db）加载美股数据
# ====================================================================

class USStockLocalLoader(DataLoader):
    """从本地 SQLite 数据库（us.db）加载美股数据。

    数据库结构
    ----------
    - ``market``     : 股票日线 OHLCV，字段 symbol/dt/open/high/low/close/volume
    - ``etf``        : ETF 日线 OHLCV，字段 symbol/dt/open/high/low/close/volume
    - ``fundamental``: 估值快照，字段 symbol/dt/market_cap/pe_ratio/pb_ratio/...
    - ``statement``  : 季度财报，字段 symbol/report_date/filing_date/revenue/net_income/...
    - ``macro``      : 宏观时间序列，字段 series_id/dt/observation_date/value/...
    - ``sp500_constituents``: S&P 500 成分股时间表，字段 symbol/dt/updated_at
    - ``industry``  : 股票行业分类映射，字段 symbol/name/sector/sub_industry/sector_etf

    Parameters
    ----------
    db_path : SQLite 数据库文件路径。
    market : 市场标识。
    filter_sp500_constituents : 兼容旧参数；当前加载流程不再用该表过滤数据。
    """

    _FUND_COL_MAP: dict[str, str] = {
        "dt":                  FundamentalCol.DATE,
        "pe_ratio":            FundamentalCol.PE,
        "pb_ratio":            FundamentalCol.PB,
        "ps_ratio":            FundamentalCol.PS,
        "roe":                 FundamentalCol.ROE,
        "roa":                 FundamentalCol.ROA,
        "gross_profit_margin": FundamentalCol.GROSS_MARGIN,
        "profit_growth_rate":  FundamentalCol.PROFIT_GROWTH,
        "debt_to_asset":       FundamentalCol.DEBT_RATIO,
        "current_ratio":       FundamentalCol.CURRENT_RATIO,
    }

    _TEMPORARILY_BLOCKED_SYMBOLS: tuple[str, ...] = ("XAUUSD",)

    def __init__(
        self,
        db_path: str | Path,
        market: str = "US"
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.market = market
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库文件不存在: {self.db_path}")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _date_bounds(
        column: str,
        start: str | date | None,
        end: str | date | None,
    ) -> tuple[list[str], list[str]]:
        conditions: list[str] = []
        params: list[str] = []
        if start is not None:
            conditions.append(f"date({column}) >= ?")
            params.append(str(pd.Timestamp(start).date()))
        if end is not None:
            conditions.append(f"date({column}) <= ?")
            params.append(str(pd.Timestamp(end).date()))
        return conditions, params

    @staticmethod
    def _preload_start(
        start: str | date | None,
        earliest: str | date | None,
    ) -> pd.Timestamp | None:
        if start is None:
            return None
        preload_start = pd.Timestamp(start).normalize() - pd.DateOffset(years=1)
        if earliest is None:
            return preload_start
        earliest_date = pd.Timestamp(earliest).normalize()
        return max(preload_start, earliest_date)

    @staticmethod
    def _preload_end(start: str | date | None) -> pd.Timestamp | None:
        if start is None:
            return None
        return pd.Timestamp(start).normalize() - pd.Timedelta(days=1)

    @staticmethod
    def _valid_preload_range(
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
    ) -> bool:
        return start is not None and end is not None and start <= end

    def _min_table_date(
        self,
        conn: sqlite3.Connection,
        table: str,
        date_expr: str,
        symbols: list[str] | None = None,
        exclude_symbols: tuple[str, ...] = (),
        extra_conditions: list[str] | None = None,
    ) -> pd.Timestamp | None:
        if not self._table_exists(conn, table):
            return None

        conditions: list[str] = []
        params: list = []
        if symbols:
            placeholders = ",".join("?" * len(symbols))
            conditions.append(f"t.symbol IN ({placeholders})")
            params.extend(symbols)
        if exclude_symbols:
            placeholders = ",".join("?" * len(exclude_symbols))
            conditions.append(f"t.symbol NOT IN ({placeholders})")
            params.extend(exclude_symbols)
        if extra_conditions:
            conditions.extend(extra_conditions)

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        row = conn.execute(
            f"SELECT MIN({date_expr}) AS min_date FROM {table} t{where}",
            params,
        ).fetchone()
        if row is None or row["min_date"] is None:
            return None
        return pd.Timestamp(row["min_date"])

    # ------------------------------------------------------------------ #
    #  行情数据
    # ------------------------------------------------------------------ #

    def load_market_data(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """加载日线行情（OHLCV）。

        Returns
        -------
        DataFrame，以 ``(date, symbol)`` 为 MultiIndex，包含
        open / high / low / close / volume / adj_close 列。
        """
        return self._load_price_table(
            table="market",
            symbols=symbols,
            start=start,
            end=end,
            exclude_symbols=self._TEMPORARILY_BLOCKED_SYMBOLS,
            log_name="USStockLocalLoader.load_market_data",
        )

    def load_etf_market_data(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """从 ``etf`` 表加载 ETF/ETP 日线行情，仍屏蔽 XAUUSD。"""
        return self._load_price_table(
            table="etf",
            symbols=symbols,
            start=start,
            end=end,
            exclude_symbols=self._TEMPORARILY_BLOCKED_SYMBOLS,
            log_name="USStockLocalLoader.load_etf_market_data",
        )

    def _load_price_table(
        self,
        table: str,
        symbols: list[str] | tuple[str, ...] | None,
        start: str | date | None,
        end: str | date | None,
        exclude_symbols: tuple[str, ...],
        log_name: str,
    ) -> pd.DataFrame:
        conditions: list[str] = []
        params: list = []

        if symbols:
            placeholders = ",".join("?" * len(symbols))
            conditions.append(f"t.symbol IN ({placeholders})")
            params.extend(symbols)
        if exclude_symbols:
            placeholders = ",".join("?" * len(exclude_symbols))
            conditions.append(f"t.symbol NOT IN ({placeholders})")
            params.extend(exclude_symbols)
        if start is not None:
            conditions.append("date(t.dt) >= ?")
            params.append(str(pd.Timestamp(start).date()))
        if end is not None:
            conditions.append("date(t.dt) <= ?")
            params.append(str(pd.Timestamp(end).date()))

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = (
            "SELECT t.symbol, t.dt, t.open, t.high, t.low, t.close, t.volume"
            f" FROM {table} t{where} ORDER BY t.dt, t.symbol"
        )

        logger.debug("{} query: {}", log_name, query)
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame()

        df = df.rename(columns={"dt": Col.DATE})
        df[Col.DATE] = pd.to_datetime(df[Col.DATE])
        df[Col.SYMBOL] = df[Col.SYMBOL].astype(str)
        df[Col.ADJ_CLOSE] = df[Col.CLOSE]
        return self._set_index(df)

    # ------------------------------------------------------------------ #
    #  基本面数据
    # ------------------------------------------------------------------ #

    def load_fundamental_data(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """从 ``fundamental`` 表加载估值快照数据。

        Returns
        -------
        DataFrame，以 ``(date, symbol)`` 为 MultiIndex，包含
        pe / pb / ps / roe / roa / gross_margin / profit_growth /
        debt_ratio / current_ratio / market_cap 列。
        """
        with self._connect() as conn:
            df = self._load_fundamental_table(conn, symbols, start, end)

        if df.empty:
            return pd.DataFrame()

        df = df.rename(columns=self._FUND_COL_MAP)
        df[FundamentalCol.DATE] = pd.to_datetime(df[FundamentalCol.DATE])
        df[FundamentalCol.SYMBOL] = df[FundamentalCol.SYMBOL].astype(str)
        return self._set_index(df)

    def _load_fundamental_table(
        self,
        conn: sqlite3.Connection,
        symbols: list[str] | None,
        start: str | date | None,
        end: str | date | None,
    ) -> pd.DataFrame:
        table = "fundamental"
        conditions: list[str] = []
        params: list = []

        if symbols:
            placeholders = ",".join("?" * len(symbols))
            conditions.append(f"t.symbol IN ({placeholders})")
            params.extend(symbols)
        if self._TEMPORARILY_BLOCKED_SYMBOLS:
            placeholders = ",".join("?" * len(self._TEMPORARILY_BLOCKED_SYMBOLS))
            conditions.append(f"t.symbol NOT IN ({placeholders})")
            params.extend(self._TEMPORARILY_BLOCKED_SYMBOLS)
        if start is not None:
            conditions.append("t.dt >= ?")
            params.append(str(pd.Timestamp(start).date()))
        if end is not None:
            conditions.append("t.dt <= ?")
            params.append(str(pd.Timestamp(end).date()))

        columns = self._table_columns(conn, table)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        wanted = [
            "symbol",
            "dt",
            "pe_ratio",
            "pb_ratio",
            "ps_ratio",
            "market_cap",
            "roe",
            "roa",
            "gross_profit_margin",
            "profit_growth_rate",
            "debt_to_asset",
            "current_ratio",
        ]
        selected = [column for column in wanted if column in columns]
        select_columns = ", ".join(f"t.{column}" for column in selected)
        query = f"SELECT {select_columns} FROM {table} t{where} ORDER BY t.dt, t.symbol"

        logger.debug("USStockLocalLoader.load_fundamental_data query: {}", query)
        return pd.read_sql_query(query, conn, params=params)

    def load_alpha158_data(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """从 ``alpha158`` 表加载预计算 Alpha158 暴露。"""
        table = "alpha158"
        conditions: list[str] = []
        params: list = []

        if symbols:
            placeholders = ",".join("?" * len(symbols))
            conditions.append(f"symbol IN ({placeholders})")
            params.extend(symbols)
        if self._TEMPORARILY_BLOCKED_SYMBOLS:
            placeholders = ",".join("?" * len(self._TEMPORARILY_BLOCKED_SYMBOLS))
            conditions.append(f"symbol NOT IN ({placeholders})")
            params.extend(self._TEMPORARILY_BLOCKED_SYMBOLS)
        if start is not None:
            conditions.append("date(dt) >= ?")
            params.append(str(pd.Timestamp(start).date()))
        if end is not None:
            conditions.append("date(dt) <= ?")
            params.append(str(pd.Timestamp(end).date()))

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        with self._connect() as conn:
            table_info = conn.execute(f"PRAGMA table_info({table})").fetchall()
            alpha_columns = [row["name"] for row in table_info if row["name"].startswith("alpha158_")]
            if not alpha_columns:
                logger.warning("alpha158 表中未找到 alpha158_ 因子列")
                return pd.DataFrame()
            selected = ["symbol", "dt", *alpha_columns]
            query = f"SELECT {', '.join(selected)} FROM {table}{where} ORDER BY dt, symbol"
            logger.debug("USStockLocalLoader.load_alpha158_data query: {}", query)
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame()

        df = df.rename(columns={"dt": Col.DATE})
        df[Col.DATE] = pd.to_datetime(df[Col.DATE])
        df[Col.SYMBOL] = df[Col.SYMBOL].astype(str)
        return self._set_index(df)

    def load_statement_data(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """从 ``statement`` 表加载财报数据，以 filing_date 延后 45 天作为 date 索引。"""
        conditions: list[str] = []
        params: list = []
        statement_delay = pd.Timedelta(days=45)
        if symbols:
            placeholders = ",".join("?" * len(symbols))
            conditions.append(f"t.symbol IN ({placeholders})")
            params.extend(symbols)
        if self._TEMPORARILY_BLOCKED_SYMBOLS:
            placeholders = ",".join("?" * len(self._TEMPORARILY_BLOCKED_SYMBOLS))
            conditions.append(f"t.symbol NOT IN ({placeholders})")
            params.extend(self._TEMPORARILY_BLOCKED_SYMBOLS)
        conditions.append("t.filing_date IS NOT NULL")
        raw_start = pd.Timestamp(start) - statement_delay if start is not None else None
        raw_end = pd.Timestamp(end) - statement_delay if end is not None else None
        date_conditions, date_params = self._date_bounds("t.filing_date", raw_start, raw_end)
        conditions.extend(date_conditions)
        params.extend(date_params)

        with self._connect() as conn:
            where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
            query = (
                "SELECT t.*, t.filing_date AS date "
                f"FROM statement t{where} ORDER BY date, t.symbol"
            )
            logger.debug("USStockLocalLoader.load_statement_data query: {}", query)
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame()

        df[Col.DATE] = pd.to_datetime(df[Col.DATE]) + statement_delay
        df[Col.SYMBOL] = df[Col.SYMBOL].astype(str)
        df = self._filter(df, symbols, start, end)
        return self._set_index(df)

    def load_macro_data(
        self,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """从 ``macro`` 表加载宏观数据，以发布日期 dt 作为 date 索引。"""
        conditions, params = self._date_bounds("dt", start, end)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = (
            "SELECT series_id, dt AS date, observation_date, value, units, updated_at "
            f"FROM macro{where} ORDER BY dt, series_id, observation_date"
        )
        logger.debug("USStockLocalLoader.load_macro_data query: {}", query)
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame()

        df[Col.DATE] = pd.to_datetime(df[Col.DATE])
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        df["series_id"] = df["series_id"].astype(str)
        return df.set_index([Col.DATE, "series_id"]).sort_index()

    def load_sp500_constituents_data(
        self,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """加载 S&P 500 成分股时间表，不参与其他数据表过滤。"""
        conditions, params = self._date_bounds("dt", start, end)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = (
            "SELECT symbol, dt AS date, updated_at "
            f"FROM sp500_constituents{where} ORDER BY dt, symbol"
        )
        logger.debug("USStockLocalLoader.load_sp500_constituents_data query: {}", query)
        with self._connect() as conn:
            if not self._table_exists(conn, "sp500_constituents"):
                logger.warning("sp500_constituents 表不存在，返回空表")
                return pd.DataFrame()
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame()

        df[Col.DATE] = pd.to_datetime(df[Col.DATE])
        df[Col.SYMBOL] = df[Col.SYMBOL].astype(str)
        if "updated_at" in df.columns:
            df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
        return self._set_index(df)

    def load_industry_data(
        self,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """加载股票行业分类静态映射表。"""
        with self._connect() as conn:
            if not self._table_exists(conn, "industry"):
                logger.warning("industry 表不存在，返回空表")
                return pd.DataFrame()

            columns = self._table_columns(conn, "industry")
            wanted = ["symbol", "name", "sector", "sub_industry", "sector_etf", "updated_at"]
            selected = [column for column in wanted if column in columns]
            if Col.SYMBOL not in selected:
                logger.warning("industry 表缺少 symbol 列，返回空表")
                return pd.DataFrame()

            conditions: list[str] = []
            params: list = []
            if symbols:
                placeholders = ",".join("?" * len(symbols))
                conditions.append(f"symbol IN ({placeholders})")
                params.extend(symbols)
            where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
            query = f"SELECT {', '.join(selected)} FROM industry{where} ORDER BY symbol"
            logger.debug("USStockLocalLoader.load_industry_data query: {}", query)
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame()

        df[Col.SYMBOL] = df[Col.SYMBOL].astype(str)
        if "updated_at" in df.columns:
            df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
        return df.set_index(Col.SYMBOL).sort_index()

    def load_data(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """一次性返回 us.db 各表数据，key 与数据库表名一致。"""
        return {
            "market": self.load_market_data(symbols, start, end),
            "etf": self.load_etf_market_data(None, start, end),
            "fundamental": self.load_fundamental_data(symbols, start, end),
            "statement": self.load_statement_data(symbols, start, end),
            "macro": self.load_macro_data(start, end),
            "sp500_constituents": self.load_sp500_constituents_data(start, end),
            "industry": self.load_industry_data(symbols),
        }

    def load_predata(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """加载 ``start`` 前一年的预热数据，不早于各表实际最早可用日期。"""
        pre_end = self._preload_end(start)
        if pre_end is None:
            return {}

        with self._connect() as conn:
            market_start = self._preload_start(
                start,
                self._min_table_date(
                    conn,
                    table="market",
                    date_expr="date(t.dt)",
                    symbols=symbols,
                    exclude_symbols=self._TEMPORARILY_BLOCKED_SYMBOLS,
                ),
            )
            etf_start = self._preload_start(
                start,
                self._min_table_date(
                    conn,
                    table="etf",
                    date_expr="date(t.dt)",
                    exclude_symbols=self._TEMPORARILY_BLOCKED_SYMBOLS,
                ),
            )
            fundamental_start = self._preload_start(
                start,
                self._min_table_date(
                    conn,
                    table="fundamental",
                    date_expr="date(t.dt)",
                    symbols=symbols,
                    exclude_symbols=self._TEMPORARILY_BLOCKED_SYMBOLS,
                ),
            )
            statement_start = self._preload_start(
                start,
                self._min_table_date(
                    conn,
                    table="statement",
                    date_expr="date(t.filing_date, '+45 days')",
                    symbols=symbols,
                    exclude_symbols=self._TEMPORARILY_BLOCKED_SYMBOLS,
                    extra_conditions=["t.filing_date IS NOT NULL"],
                ),
            )
            macro_start = self._preload_start(
                start,
                self._min_table_date(conn, table="macro", date_expr="date(t.dt)"),
            )
            sp500_start = self._preload_start(
                start,
                self._min_table_date(conn, table="sp500_constituents", date_expr="date(t.dt)"),
            )

        return {
            "market": (
                self.load_market_data(symbols, market_start, pre_end)
                if self._valid_preload_range(market_start, pre_end)
                else pd.DataFrame()
            ),
            "etf": (
                self.load_etf_market_data(None, etf_start, pre_end)
                if self._valid_preload_range(etf_start, pre_end)
                else pd.DataFrame()
            ),
            "fundamental": (
                self.load_fundamental_data(symbols, fundamental_start, pre_end)
                if self._valid_preload_range(fundamental_start, pre_end)
                else pd.DataFrame()
            ),
            "statement": (
                self.load_statement_data(symbols, statement_start, pre_end)
                if self._valid_preload_range(statement_start, pre_end)
                else pd.DataFrame()
            ),
            "macro": (
                self.load_macro_data(macro_start, pre_end)
                if self._valid_preload_range(macro_start, pre_end)
                else pd.DataFrame()
            ),
            "sp500_constituents": (
                self.load_sp500_constituents_data(sp500_start, pre_end)
                if self._valid_preload_range(sp500_start, pre_end)
                else pd.DataFrame()
            )
        }
