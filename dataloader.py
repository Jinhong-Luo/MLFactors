"""读取按股票拆分的数据和每日 S&P 500 成分股。"""

from datetime import date
from pathlib import Path

import pandas as pd


def load_data(
    symbols: list[str] | None = None,
    start: str | date | None = None,
    end: str | date | None = None,
    data_dir: str | Path = Path(__file__).resolve().parent / "cache" / "csv",
    constituents_path: str | Path | None = None,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """返回 (date, symbol) MultiIndex 数据和每日成分股字典。"""
    data_dir = Path(data_dir).expanduser().resolve()

    if symbols is None:
        csv_files = sorted(data_dir.glob("*.csv"))
    else:
        csv_files = [data_dir / f"{symbol}.csv" for symbol in symbols]

    frames = []
    for csv_file in csv_files:
        usecols = ["date", *columns] if columns is not None else None
        frame = pd.read_csv(csv_file, usecols=usecols, parse_dates=["date"])
        frame["symbol"] = csv_file.stem
        if start is not None:
            frame = frame[frame["date"] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame["date"] <= pd.Timestamp(end)]
        frames.append(frame)

    if frames:
        data = pd.concat(frames, ignore_index=True)
        data = data.set_index(["date", "symbol"]).sort_index()
    else:
        index = pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])
        data = pd.DataFrame(index=index)

    constituent_dict = {}
    if constituents_path is not None:
        constituents_path = Path(constituents_path).expanduser().resolve()
        constituents = pd.read_csv(constituents_path, dtype=str)
        if start is not None:
            constituents = constituents[constituents["date"] >= str(pd.Timestamp(start).date())]
        if end is not None:
            constituents = constituents[constituents["date"] <= str(pd.Timestamp(end).date())]
        constituent_dict = {
            row.date: set(row.tickers.split(","))
            for row in constituents.itertuples(index=False)
        }

    return data, constituent_dict
