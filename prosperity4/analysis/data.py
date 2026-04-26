from pathlib import Path
from typing import cast

import pandas as pd

#all of these are global ugly as hell utility functions womp womp

def _tutorial_round_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "tutorial_round"


def _round1_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "ROUND_1"


def _round2_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "ROUND_2"


def _round3_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "ROUND_3"


def _round4_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "ROUND_4"



def _read_csv(path: Path) -> pd.DataFrame:
    return cast(pd.DataFrame, pd.read_csv(path, sep=";"))



def read_tutorial_prices(day: int) -> pd.DataFrame:
    path = _tutorial_round_dir() / f"prices_round_0_day_{day}.csv"
    return _read_csv(path)

def read_tutorial_trades(day: int) -> pd.DataFrame:
    path = _tutorial_round_dir() / f"trades_round_0_day_{day}.csv"
    return _read_csv(path)

def read_all_tutorial_prices() -> pd.DataFrame:
    base = _tutorial_round_dir()
    frames: list[pd.DataFrame] = [_read_csv(path) for path in sorted(base.glob("prices_round_0_day_*.csv"))]
    if not frames:
        return pd.DataFrame()
    return cast(pd.DataFrame, pd.concat(frames, ignore_index=True))

def read_all_tutorial_trades() -> pd.DataFrame:
    base = _tutorial_round_dir()
    frames: list[pd.DataFrame] = [_read_csv(path) for path in sorted(base.glob("trades_round_0_day_*.csv"))]
    if not frames:
        return pd.DataFrame()
    return cast(pd.DataFrame, pd.concat(frames, ignore_index=True))


def read_round1_prices(day: int) -> pd.DataFrame:
    path = _round1_dir() / f"prices_round_1_day_{day}.csv"
    return _read_csv(path)

def read_round1_trades(day: int) -> pd.DataFrame:
    path = _round1_dir() / f"trades_round_1_day_{day}.csv"
    return _read_csv(path)

def read_all_round1_prices() -> pd.DataFrame:
    base = _round2_dir()
    frames: list[pd.DataFrame] = [_read_csv(path) for path in sorted(base.glob("prices_round_2_day_*.csv"))]
    if not frames:
        return pd.DataFrame()
    return cast(pd.DataFrame, pd.concat(frames, ignore_index=True))

def read_all_round1_trades() -> pd.DataFrame:
    base = _round1_dir()
    frames: list[pd.DataFrame] = []
    for path in sorted(base.glob("trades_round_1_day_*.csv")):
        frame = _read_csv(path)
        # Trades files do not carry day in schema, so derive it from filename.
        day = int(path.stem.split("day_")[-1])
        frame["day"] = day
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return cast(pd.DataFrame, pd.concat(frames, ignore_index=True))


def read_round2_prices(day: int) -> pd.DataFrame:
    path = _round2_dir() / f"prices_round_2_day_{day}.csv"
    return _read_csv(path)

def read_round2_trades(day: int) -> pd.DataFrame:
    path = _round2_dir() / f"trades_round_2_day_{day}.csv"
    return _read_csv(path)

def read_all_round2_prices() -> pd.DataFrame:
    base = _round2_dir()
    frames: list[pd.DataFrame] = [_read_csv(path) for path in sorted(base.glob("prices_round_2_day_*.csv"))]
    if not frames:
        return pd.DataFrame()
    return cast(pd.DataFrame, pd.concat(frames, ignore_index=True))

def read_all_round2_trades() -> pd.DataFrame:
    base = _round1_dir()
    frames: list[pd.DataFrame] = []
    for path in sorted(base.glob("trades_round_2_day_*.csv")):
        frame = _read_csv(path)
        # Trades files do not carry day in schema, so derive it from filename.
        day = int(path.stem.split("day_")[-1])
        frame["day"] = day
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return cast(pd.DataFrame, pd.concat(frames, ignore_index=True))


def read_round3_prices(day: int) -> pd.DataFrame:
    path = _round3_dir() / f"prices_round_3_day_{day}.csv"
    return _read_csv(path)

def read_round3_trades(day: int) -> pd.DataFrame:
    path = _round3_dir() / f"trades_round_3_day_{day}.csv"
    return _read_csv(path)

def read_all_round3_prices() -> pd.DataFrame:
    base = _round3_dir()
    frames: list[pd.DataFrame] = [_read_csv(path) for path in sorted(base.glob("prices_round_3_day_*.csv"))]
    if not frames:
        return pd.DataFrame()
    return cast(pd.DataFrame, pd.concat(frames, ignore_index=True))

def read_all_round3_trades() -> pd.DataFrame:
    base = _round3_dir()
    frames: list[pd.DataFrame] = []
    for path in sorted(base.glob("trades_round_3_day_*.csv")):
        frame = _read_csv(path)
        # Trades files do not carry day in schema, so derive it from filename.
        day = int(path.stem.split("day_")[-1])
        frame["day"] = day
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return cast(pd.DataFrame, pd.concat(frames, ignore_index=True))


def read_all_round4_prices() -> pd.DataFrame:
    base = _round4_dir()
    frames: list[pd.DataFrame] = [_read_csv(path) for path in sorted(base.glob("prices_round_4_day_*.csv"))]
    if not frames:
        return pd.DataFrame()
    return cast(pd.DataFrame, pd.concat(frames, ignore_index=True))
