"""Repo-local readers used by principal adapter datasets."""

from __future__ import annotations

import logging
from datetime import datetime
from io import StringIO
from os import PathLike
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from sciona.principal.adapters.polar import PolarH10DataSet
from sciona.principal.datasets.core import msg

log = logging.getLogger(__name__)


CAPNO_SKIP_ROWS = 9
CAPNO_COLS = [
    "DATE",
    "TIME",
    "CO2 Wave",
    "EtCO2",
    "FiCO2",
    "RR",
    "SpO2",
    "PR",
    "IPI",
    "A/hr",
    "ODI",
    "ETCO2 HIGH URGENT ALARM",
    "ETCO2 LOW URGENT ALARM",
    "ETCO2 HIGH CAUTION ALARM",
    "ETCO2 LOW CAUTION ALARM",
    "FICO2 HIGH URGENT ALARM",
    "FICO2 HIGH CAUTION ALARM",
    "RR HIGH URGENT ALARM",
    "RR LOW URGENT ALARM",
    "RR HIGH CAUTION ALARM",
    "RR LOW CAUTION ALARM",
    "NO BREATH ALARM",
    "SPO2 HIGH URGENT ALARM",
    "SPO2 LOW URGENT  ALARM",
    "SPO2 HIGH CAUTION ALARM",
    "SPO2 LOW CAUTION ALARM",
    "PR HIGH URGENT ALARM",
    "PR LOW URGENT ALARM",
    "PR HIGH CAUTION ALARM",
    "PR LOW CAUTION ALARM",
    "IPI LOW URGENT ALARM",
    "IPI LOW CAUTION ALARM",
    "APNEA EVENT ≥ 10 sec",
    "APNEA EVENT 10-19 sec",
    "APNEA EVENT 20-29 sec",
    "APNEA EVENT ≥ 30 sec",
    "DESAT EVENT",
    "CO2 NOT AVAILABLE",
    "SPO2 NOT AVAILABLE",
    "BATTERY LOW",
    "EVENT 1",
    "EVENT 2",
    "EVENT 3",
]
CAPNO_KEEP_COLS = [
    "DATE",
    "TIME",
    "CO2 Wave",
    "EtCO2",
    "FiCO2",
    "RR",
    "SpO2",
    "PR",
    "IPI",
    "A/hr",
    "ODI",
    "APNEA EVENT ≥ 10 sec",
    "APNEA EVENT 10-19 sec",
    "APNEA EVENT 20-29 sec",
    "APNEA EVENT ≥ 30 sec",
    "DESAT EVENT",
    "CO2 NOT AVAILABLE",
    "SPO2 NOT AVAILABLE",
    "BATTERY LOW",
    "EVENT 1",
    "EVENT 2",
    "EVENT 3",
]


def _empty_capnostream_df() -> pd.DataFrame:
    return pd.DataFrame([], columns=[*CAPNO_KEEP_COLS, "etime"])


def _meta_value(meta: Any, key: str, default: Any = None) -> Any:
    if meta is None:
        return default
    if isinstance(meta, dict):
        if key in meta:
            return meta[key]
        metadata = meta.get("metadata")
        if isinstance(metadata, dict):
            return metadata.get(key, default)
        return default
    if hasattr(meta, key):
        return getattr(meta, key)
    metadata = getattr(meta, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return default


def _meta_timezone(meta: Any) -> str:
    tz = _meta_value(meta, "tz") or _meta_value(meta, "timezone")
    if tz:
        return str(tz)
    msg.warn("Capnostream parser did not receive metadata timezone; defaulting to UTC.")
    return "UTC"


def polar_h10_to_pandas(
    filepath: PathLike,
    serial: str | None = None,
    serial_attr: str | None = None,
    only: str | list[str] | None = None,
    exclude: str | list[str] | None = None,
    single: bool = False,
    meta: object | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> pd.DataFrame | Mapping[str, pd.DataFrame]:
    """Parse a Polar H10 sqlite file into one or more pandas dataframes."""
    del kwargs
    if serial is None and serial_attr is None:
        raise ValueError("You must specify either `serial` or `serial_attr` and metadata.")

    if serial is None:
        serial = _meta_value(meta, serial_attr)

    ds = PolarH10DataSet.from_file(filepath)
    if isinstance(ds, dict):
        ds = ds.get(serial)

    if ds is None:
        return pd.DataFrame([])

    result = ds.to_pandas(
        private=False,
        exclude=["*_start", "*_stop"],
        group_only=only,
        group_exclude=exclude,
    )
    if len(result) == 1 and single:
        return next(iter(result.values()))
    return result


def parse_capnostream_folder(
    target: PathLike, meta: object | Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """Parse all Capnostream text files in a folder into a single dataframe."""
    target = Path(target).expanduser()
    all_streams: list[pd.DataFrame] = []
    skipped = 0

    for entry in sorted(target.glob("*.txt")):
        try:
            parsed = parse_capnostream_file(entry, meta=meta)
        except Exception:
            skipped += 1
            log.debug("Skipped capnostream file %s due to parser error.", entry, exc_info=True)
            continue
        if len(parsed) > 0:
            all_streams.append(parsed)

    if skipped > 0:
        msg.warn(f"While parsing Capnostream from {target}; {skipped} files caused parser errors.")
        log.warning("While parsing Capnostream from %s; %s files caused parser errors.", target, skipped)

    if not all_streams:
        return _empty_capnostream_df()

    if len(all_streams) == 1:
        return all_streams[0].sort_values("etime").reset_index(drop=True)

    sorted_streams = sorted(all_streams, key=lambda frame: frame["etime"].iloc[0])
    return pd.concat(sorted_streams, ignore_index=True).sort_values("etime").reset_index(drop=True)


def parse_capnostream_file(
    target: PathLike, meta: object | Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """Parse a single Capnostream text export into a dataframe."""
    csv_file_arr = []
    with open(target, "r", encoding="utf-16") as handle:
        for line in handle:
            csv_file_arr.append(line.replace("\0", ""))
    csv_file_str = "".join(csv_file_arr)

    try:
        capno_df = pd.read_csv(
            StringIO(csv_file_str),
            sep="\t",
            skiprows=CAPNO_SKIP_ROWS,
            header=None,
            names=CAPNO_COLS,
            engine="python",
        )
    except Exception:
        msg.err(f"While parsing {target} for Capnostream; returning empty dataframe.")
        log.error("While parsing %s for Capnostream; returning empty dataframe.", target, exc_info=True)
        return _empty_capnostream_df()

    simple_capno_df = capno_df[CAPNO_KEEP_COLS].copy()
    if simple_capno_df.empty:
        return _empty_capnostream_df()

    first_time = simple_capno_df["TIME"].iloc[0]
    last_time = simple_capno_df["TIME"].iloc[-1]

    for idx in range(simple_capno_df.shape[0]):
        if simple_capno_df["TIME"].iloc[idx] != first_time:
            simple_capno_df = simple_capno_df.drop(simple_capno_df.index[:idx])
            break
    simple_capno_df = simple_capno_df.reset_index(drop=True)

    for idx in range(simple_capno_df.shape[0] - 1, 0, -1):
        if simple_capno_df["TIME"].iloc[idx] != last_time:
            if simple_capno_df.shape[0] != idx + 1:
                simple_capno_df = simple_capno_df.drop(simple_capno_df.index[idx + 1 :])
            break
    simple_capno_df = simple_capno_df.reset_index(drop=True)

    def _fix_clock(ts: Any) -> Any:
        if not isinstance(ts, str):
            return ts
        parts = ts.split(":")
        if parts and parts[0] in {"0", "00"}:
            parts[0] = "12"
            return ":".join(parts)
        return ts

    simple_capno_df["TIME"] = simple_capno_df["TIME"].map(_fix_clock)

    combined = simple_capno_df["DATE"].astype(str) + " " + simple_capno_df["TIME"].astype(str)
    naive = pd.to_datetime(combined, format="%b %d, %y %I:%M:%S %p", errors="coerce")
    simple_capno_df = simple_capno_df.loc[naive.notna()].copy()
    if simple_capno_df.empty:
        return _empty_capnostream_df()

    tz_name = _meta_timezone(meta)
    localized = naive.loc[naive.notna()].dt.tz_localize(
        tz_name, ambiguous="NaT", nonexistent="shift_forward"
    )
    simple_capno_df = simple_capno_df.loc[localized.notna()].copy()
    localized = localized.loc[localized.notna()]
    if simple_capno_df.empty:
        return _empty_capnostream_df()

    simple_capno_df["datetime"] = localized
    simple_capno_df = simple_capno_df.sort_values("datetime").reset_index(drop=True)
    epoch_seconds = simple_capno_df["datetime"].astype("int64").to_numpy() / 1e9
    unique_seconds = np.unique(epoch_seconds)

    if len(unique_seconds) < 3:
        return _empty_capnostream_df()

    adjusted: list[np.ndarray] = []
    for idx in range(1, len(unique_seconds) - 1):
        current = unique_seconds[idx]
        count = int(np.sum(epoch_seconds == current))
        end_ts = unique_seconds[idx + 1]
        adjusted.append(np.linspace(current, end_ts - (end_ts - current) / count, count))

    keep_mask = (epoch_seconds > unique_seconds[0]) & (epoch_seconds < unique_seconds[-1])
    simple_capno_df = simple_capno_df.loc[keep_mask].copy().reset_index(drop=True)
    if not adjusted or simple_capno_df.empty:
        return _empty_capnostream_df()

    simple_capno_df["etime"] = np.hstack(adjusted)
    return simple_capno_df.drop(columns=["datetime"])
