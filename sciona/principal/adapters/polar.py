"""Trimmed Polar H10 dataset reader for principal adapter datasets."""

from __future__ import annotations

import logging
import sqlite3
import struct
from dataclasses import dataclass, field
from enum import Enum
from os import PathLike, path
from pathlib import Path
from typing import Mapping

import numpy as np

from sciona.principal.datasets._parser import DataSetBase, ResamplingOptions
from sciona.principal.datasets.core import msg, run_vector_property

log = logging.getLogger(__name__)


class PolarDataKind(str, Enum):
    ECG = "ecg"
    ACC = "acc"
    HR = "hr"


NS_PER_S = 1e9
ECG_NORMAL_FRAME_TICKS_DELT = 560697041
ECG_SAMPLES_PER_FRAME = 73
ECG_NOMINAL_SLOPE = ECG_NORMAL_FRAME_TICKS_DELT / ECG_SAMPLES_PER_FRAME
ACC_NORMAL_FRAME_TICKS_DELT = 1409533952
ACC_SAMPLES_PER_FRAME = 36
ACC_NOMINAL_SLOPE = ACC_NORMAL_FRAME_TICKS_DELT / ACC_SAMPLES_PER_FRAME
ACC_INT_TO_GS = 1000.0
HR_TICKS_VAL = 1000

ID_COL = 0
TS_COL = 1
DEV_ID_COL = 2
INFO_DEV_NAME_COL = 3
DEV_TS_COL = 3
ECG_CSV_COL = 4
ACC_DEV_TS_COL = 3
ACC_CSV_X_COL = 4
ACC_CSV_Y_COL = 5
ACC_CSV_Z_COL = 6
HR_BPM_COL = 3
HR_RR_COL = 4

MIN_DATA_LEN = {
    PolarDataKind.HR: 200,
}
NOMINAL_SLOPE = {
    PolarDataKind.ECG: ECG_NOMINAL_SLOPE,
    PolarDataKind.ACC: ACC_NOMINAL_SLOPE,
    PolarDataKind.HR: 1,
}


def _sql_limit_str(limit: int | None) -> str:
    return f"LIMIT {limit}" if limit is not None else ""


def _has_acc_data(cursor: sqlite3.Cursor) -> bool:
    cursor.execute(
        "SELECT count(name) FROM sqlite_master WHERE type='table' AND name='acceleration'"
    )
    row = cursor.fetchone()
    return bool(row and row[0] == 1)


def _parse_any_acc_csv(acc_csv: str) -> list[float]:
    values = [int(val) / ACC_INT_TO_GS for val in acc_csv.split(",")]
    if len(values) != ACC_SAMPLES_PER_FRAME:
        log.debug("Abnormal acc len=%s %s", len(values), values[:5])
    return values


def process_ecg_sample(
    t0: float, tslope: float, samples: dict[str, object]
) -> list[tuple[float, float]]:
    return [
        (t0 + tslope * i, val) for i, val in enumerate(samples["Samples"])  # type: ignore[index]
    ]


def process_acc_sample(
    t0: float, tslope: float, samples: dict[str, object]
) -> list[tuple[float, float, float, float]]:
    return [
        (t0 + tslope * i, *val)
        for i, val in enumerate(
            zip(samples["x"], samples["y"], samples["z"])  # type: ignore[index]
        )
    ]


def process_hr_sample(
    t0: float, tslope: float, samples: dict[str, object]
) -> list[tuple[float, float, float]]:
    result: list[tuple[float, float, float]] = []
    rr_count = 0.0
    hr = float(samples["hr"])  # type: ignore[index]
    for val in samples["Samples"]:  # type: ignore[index]
        result.append((t0 + (tslope * rr_count), float(val), hr))
        rr_count += float(val) / HR_TICKS_VAL * NS_PER_S
    return result


def default_t0(
    host_t0: float | None = None, tslope: float | None = None, count: int | None = None, **kwargs
) -> float:
    del kwargs
    assert host_t0 is not None and tslope is not None and count is not None
    return host_t0 + tslope * count


def hr_t0(
    host_t0: float | None = None,
    tslope: float | None = None,
    dev_t0: float | None = None,
    dev_ts: float | None = None,
    **kwargs,
) -> float:
    del kwargs
    assert host_t0 is not None and tslope is not None and dev_t0 is not None and dev_ts is not None
    return host_t0 + tslope * (dev_ts - dev_t0)


def _get_ecg_dev_samples(
    cursor: sqlite3.Cursor, device_id: str, limit: int | None = None
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    limit_str = _sql_limit_str(limit)
    for row in cursor.execute(
        f"SELECT * FROM ecg WHERE device_id = {device_id} {limit_str}"
    ):
        ecg_ints = [int(val) for val in row[ECG_CSV_COL].split(",")]
        result.append(
            {
                "host_ts": row[TS_COL],
                "Timestamp": row[DEV_TS_COL],
                "Samples": ecg_ints,
            }
        )
    return result


def _get_acc_dev_samples(
    cursor: sqlite3.Cursor, device_id: str, limit: int | None = None
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    if not _has_acc_data(cursor):
        return result
    limit_str = _sql_limit_str(limit)
    for row in cursor.execute(
        f"SELECT * FROM acceleration WHERE device_id = {device_id} {limit_str}"
    ):
        result.append(
            {
                "host_ts": row[TS_COL],
                "Timestamp": row[ACC_DEV_TS_COL],
                "x": _parse_any_acc_csv(row[ACC_CSV_X_COL]),
                "y": _parse_any_acc_csv(row[ACC_CSV_Y_COL]),
                "z": _parse_any_acc_csv(row[ACC_CSV_Z_COL]),
            }
        )
    return result


def _get_hr_dev_samples(
    cursor: sqlite3.Cursor, device_id: str, limit: int | None = None
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    t0 = 0.0
    rr_sum = 0.0
    limit_str = _sql_limit_str(limit)
    for row in cursor.execute(
        f"SELECT * FROM heart_rate WHERE device_id = {device_id} {limit_str}"
    ):
        cur_ts = row[TS_COL]
        rr_csv = row[HR_RR_COL]
        if rr_sum == 0:
            t0 = cur_ts
        dev_ts = t0 + (rr_sum / HR_TICKS_VAL) * NS_PER_S
        rr_ints: list[int] = []
        if len(rr_csv) > 0:
            rr_ints = [int(val[:-2]) for val in rr_csv.split(",")]
        rr_sum += sum(rr_ints)
        result.append(
            {
                "host_ts": cur_ts,
                "Timestamp": dev_ts,
                "hr": row[HR_BPM_COL],
                "Samples": rr_ints,
            }
        )
    return result


def _get_any_mx_time(
    cursor: sqlite3.Cursor, device_id: str, table: str, op: str
) -> float | None:
    for row in cursor.execute(
        f"SELECT {op}(timestamp) FROM {table} WHERE device_id = {device_id}"
    ):
        value = row[0]
        if value is not None:
            return value / 1e9
    return None


def _get_hr_min_time(cursor: sqlite3.Cursor, device_id: str) -> float | None:
    return _get_any_mx_time(cursor, device_id, "heart_rate", "MIN")


def _get_acc_min_time(cursor: sqlite3.Cursor, device_id: str) -> float | None:
    return _get_any_mx_time(cursor, device_id, "acceleration", "MIN")


def _get_ecg_min_time(cursor: sqlite3.Cursor, device_id: str) -> float | None:
    return _get_any_mx_time(cursor, device_id, "ecg", "MIN")


def _get_hr_max_time(cursor: sqlite3.Cursor, device_id: str) -> float | None:
    return _get_any_mx_time(cursor, device_id, "heart_rate", "MAX")


def _get_acc_max_time(cursor: sqlite3.Cursor, device_id: str) -> float | None:
    return _get_any_mx_time(cursor, device_id, "acceleration", "MAX")


def _get_ecg_max_time(cursor: sqlite3.Cursor, device_id: str) -> float | None:
    return _get_any_mx_time(cursor, device_id, "ecg", "MAX")


def _get_ecg_sample_count(cursor: sqlite3.Cursor, device_id: str) -> int:
    for row in cursor.execute(f"SELECT COUNT(*) FROM ecg WHERE device_id = {device_id}"):
        return row[0]
    return 0


def _get_acc_sample_count(cursor: sqlite3.Cursor, device_id: str) -> int:
    if not _has_acc_data(cursor):
        return 0
    for row in cursor.execute(
        f"SELECT COUNT(*) FROM acceleration WHERE device_id = {device_id}"
    ):
        return row[0]
    return 0


def _get_hr_sample_count(cursor: sqlite3.Cursor, device_id: str) -> int:
    for row in cursor.execute(
        f"SELECT COUNT(*) FROM heart_rate WHERE device_id = {device_id}"
    ):
        return row[0]
    return 0


SAMPLE_PROCESSORS = {
    PolarDataKind.ECG: process_ecg_sample,
    PolarDataKind.ACC: process_acc_sample,
    PolarDataKind.HR: process_hr_sample,
}
T0_CALCULATORS = {
    PolarDataKind.HR: hr_t0,
}
SAMPLE_COMPILERS = {
    PolarDataKind.ECG: _get_ecg_dev_samples,
    PolarDataKind.ACC: _get_acc_dev_samples,
    PolarDataKind.HR: _get_hr_dev_samples,
}
REPLACE_NANS = [
    PolarDataKind.ECG,
]
MIN_TIME_GETTERS = {
    PolarDataKind.ECG: _get_ecg_min_time,
    PolarDataKind.ACC: _get_acc_min_time,
    PolarDataKind.HR: _get_hr_min_time,
}
MAX_TIME_GETTERS = {
    PolarDataKind.ECG: _get_ecg_max_time,
    PolarDataKind.ACC: _get_acc_max_time,
    PolarDataKind.HR: _get_hr_max_time,
}
SAMPLE_COUNT_GETTERS = {
    PolarDataKind.ECG: _get_ecg_sample_count,
    PolarDataKind.ACC: _get_acc_sample_count,
    PolarDataKind.HR: _get_hr_sample_count,
}


def process_time_correction(
    dev_id: str, data_in: list[dict[str, object]], kind: PolarDataKind
) -> list[tuple]:
    del dev_id
    if len(data_in) <= MIN_DATA_LEN.get(kind, 2):
        return []

    host_ts_arr = np.array([datum["host_ts"] for datum in data_in], dtype=float)
    host_ts_diff = np.diff(host_ts_arr)
    split_idx = np.where(host_ts_diff > 3_000_000_000)[0] + 1
    data_segments = np.split(np.array(data_in, dtype=object), split_idx)

    output: list[tuple] = []
    for segment in data_segments:
        data = segment.tolist()
        if len(data) <= MIN_DATA_LEN.get(kind, 2):
            continue
        first, last = data[0], data[-1]
        host_ts = float(first["host_ts"])
        host_te = float(last["host_ts"])
        dev_ts = float(first["Timestamp"])
        dev_te = float(last["Timestamp"])
        dhost = host_te - host_ts
        ddev = dev_te - dev_ts
        if ddev == 0 or dhost == 0:
            continue
        if kind == PolarDataKind.HR:
            tslope = (dhost / ddev) * NOMINAL_SLOPE[kind]
        else:
            samp_per_frame = 1
            if kind == PolarDataKind.ECG:
                samp_per_frame = ECG_SAMPLES_PER_FRAME
            elif kind == PolarDataKind.ACC:
                samp_per_frame = ACC_SAMPLES_PER_FRAME
            tslope = dhost / (len(data) * samp_per_frame)

        count = 0
        t0_calc = T0_CALCULATORS.get(kind, default_t0)
        for sample in data:
            t0 = t0_calc(
                host_t0=host_ts,
                dev_t0=dev_ts,
                tslope=tslope,
                count=count,
                dev_ts=float(sample["Timestamp"]),
                host_ts=float(sample["host_ts"]),
            )
            corrected = SAMPLE_PROCESSORS[kind](t0, tslope, sample)
            output.extend(corrected)
            count += len(corrected)
    return output


@dataclass
class PolarH10Raw:
    filepath: PathLike
    cursor: sqlite3.Cursor
    dev_id: str
    serial: str
    name: str
    data: Mapping[PolarDataKind, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._min: float | None = None
        self._max: float | None = None

    @property
    def has_data(self) -> bool:
        for kind in PolarDataKind:
            data = SAMPLE_COMPILERS[kind](self.cursor, device_id=self.dev_id, limit=1)
            if len(data) > 0:
                return True
        return False

    @property
    def min(self) -> float | None:
        if self._min is None:
            value = None
            for kind in PolarDataKind:
                mt = MIN_TIME_GETTERS[kind](self.cursor, self.dev_id)
                if value is None or (mt is not None and mt < value):
                    value = mt
            self._min = value
        return self._min

    @property
    def max(self) -> float | None:
        if self._max is None:
            value = None
            for kind in PolarDataKind:
                mt = MAX_TIME_GETTERS[kind](self.cursor, self.dev_id)
                if value is None or (mt is not None and mt > value):
                    value = mt
            self._max = value
        return self._max

    def _load_time_kind(self, kind: PolarDataKind) -> None:
        if kind in self.data:
            return
        data = SAMPLE_COMPILERS[kind](self.cursor, device_id=self.dev_id)
        processed = process_time_correction(self.dev_id, data, kind)
        arr = np.array(processed)
        if kind in REPLACE_NANS and len(arr) > 0:
            arr[:, 1:] = np.nan_to_num(arr[:, 1:])
        self.data[kind] = arr

    def get(self, kind: PolarDataKind, columns: tuple[int, ...]) -> np.ndarray:
        self._load_time_kind(kind)
        data = self.data.get(kind)
        if data is None or len(data) == 0:
            if len(columns) > 1:
                return np.array([[] for _ in range(len(columns))]).T
            return np.array([])
        return data[:, columns]


def parse_polar_sqlite(
    source: PathLike, serial: str | None = None
) -> Mapping[str, list[PolarH10Raw]] | list[PolarH10Raw]:
    target = Path(source).expanduser()
    if not target.exists():
        return {}

    con = sqlite3.connect(target)
    cur = con.cursor()
    result: dict[str, list[PolarH10Raw]] = {}
    selected_serial: str | None = None

    for row in cur.execute("SELECT * FROM device"):
        dev_id = row[ID_COL]
        dev_name = row[INFO_DEV_NAME_COL]
        if not isinstance(dev_name, str):
            continue
        device_serial = dev_name.split()[-1]
        if serial is not None and serial not in dev_name:
            continue
        result.setdefault(device_serial, []).append(
            PolarH10Raw(target, cur, str(dev_id), device_serial, dev_name)
        )
        selected_serial = device_serial

    if not result:
        msg.warn(f"The file {source} has no devices matching serial {serial}.")
        return {}

    if serial is not None:
        return result.get(selected_serial or "", [])
    return result


class PolarH10DataSet(DataSetBase):
    """Single-dataset view over Polar H10 ECG/HR/IBI/acceleration data."""

    TIME_SERIES = ["hr_t", "ibi_t", "ecg_t", "accel_t"]

    def __init__(self, raw: PolarH10Raw | list[PolarH10Raw] | None) -> None:
        super().__init__()
        if raw is None:
            self.raw: list[PolarH10Raw] = []
        elif isinstance(raw, list):
            self.raw = raw
        else:
            self.raw = [raw]

        self._ibi = None
        self._ibi_t = None
        self._ecg = None
        self._ecg_t = None
        self._accel_x = None
        self._accel_y = None
        self._accel_z = None
        self._accel_t = None
        self._hr = None
        self._hr_t = None

    @classmethod
    def empty(cls) -> "PolarH10DataSet":
        return cls(None)

    def __len__(self) -> int:
        total = 0
        for kind in PolarDataKind:
            for raw in self.raw:
                total += SAMPLE_COUNT_GETTERS[kind](raw.cursor, raw.dev_id)
        return total

    def lazy_load_attrs(self) -> None:
        for kind in PolarDataKind:
            self._lazy_load(kind)

    @property
    def min(self) -> float:
        if self.raw:
            vals = [raw.min for raw in self.raw if raw.min is not None]
            if vals:
                return min(vals)
        return super().min

    @property
    def max(self) -> float:
        if self.raw:
            vals = [raw.max for raw in self.raw if raw.max is not None]
            if vals:
                return max(vals)
        return super().max

    def _lazy_load(self, kind: PolarDataKind) -> None:
        if self.raw:
            if kind == PolarDataKind.HR and self._hr is None:
                hr, hrt, ibi, ibit = [], [], [], []
                for raw in self.raw:
                    hrt_raw = raw.get(PolarDataKind.HR, (0,)).flatten()
                    if len(hrt_raw) > 0:
                        ibi.append(np.concatenate(([hrt_raw[0]], np.diff(hrt_raw))) / 1e9)
                        ibit.append(hrt_raw / 1e9)
                    else:
                        ibi.append(np.array([]))
                        ibit.append(np.array([]))
                    hr.append(raw.get(PolarDataKind.HR, (2,)).flatten())
                    hrt.append(hrt_raw / 1e9)
                self._hr = np.concatenate(hr) if hr else np.array([])
                self._hr_t = np.concatenate(hrt) if hrt else np.array([])
                self._ibi = np.concatenate(ibi) if ibi else np.array([])
                self._ibi_t = np.concatenate(ibit) if ibit else np.array([])
            elif kind == PolarDataKind.ECG and self._ecg is None:
                ecg, ecg_t = [], []
                for raw in self.raw:
                    ecg.append(raw.get(PolarDataKind.ECG, (1,)).flatten())
                    ecg_t.append(raw.get(PolarDataKind.ECG, (0,)).flatten() / 1e9)
                self._ecg = np.concatenate(ecg) if ecg else np.array([])
                self._ecg_t = np.concatenate(ecg_t) if ecg_t else np.array([])
            elif kind == PolarDataKind.ACC and self._accel_x is None:
                accel_x, accel_y, accel_z, accel_t = [], [], [], []
                for raw in self.raw:
                    accel_x.append(raw.get(PolarDataKind.ACC, (1,)).flatten())
                    accel_y.append(raw.get(PolarDataKind.ACC, (2,)).flatten())
                    accel_z.append(raw.get(PolarDataKind.ACC, (3,)).flatten())
                    accel_t.append(raw.get(PolarDataKind.ACC, (0,)).flatten() / 1e9)
                self._accel_x = np.concatenate(accel_x) if accel_x else np.array([])
                self._accel_y = np.concatenate(accel_y) if accel_y else np.array([])
                self._accel_z = np.concatenate(accel_z) if accel_z else np.array([])
                self._accel_t = np.concatenate(accel_t) if accel_t else np.array([])
            return

        if kind == PolarDataKind.HR and self._ibi is None:
            self._ibi = np.array([])
            self._ibi_t = np.array([])
            self._hr = np.array([])
            self._hr_t = np.array([])
        elif kind == PolarDataKind.ECG and self._ecg is None:
            self._ecg = np.array([])
            self._ecg_t = np.array([])
        elif kind == PolarDataKind.ACC and self._accel_x is None:
            self._accel_x = np.array([])
            self._accel_y = np.array([])
            self._accel_z = np.array([])
            self._accel_t = np.array([])

    @property
    def ibi(self) -> np.ndarray:
        if self._ibi is None:
            self._lazy_load(PolarDataKind.HR)
        return run_vector_property(
            "ibi", "ibi_t", self._cast, self._ibi, self._ibi_t, self.hr_t_start, self.hr_t_stop, self.resampling
        )

    @property
    def ibi_t(self) -> np.ndarray:
        if self._ibi_t is None:
            self._lazy_load(PolarDataKind.HR)
        return run_vector_property(
            "ibi_t", "ibi_t", self._cast, self._ibi_t, self._ibi_t, self.hr_t_start, self.hr_t_stop, self.resampling
        )

    @property
    def hr(self) -> np.ndarray:
        if self._hr is None:
            self._lazy_load(PolarDataKind.HR)
        return run_vector_property(
            "hr", "hr_t", self._cast, self._hr, self._hr_t, self.hr_t_start, self.hr_t_stop, self.resampling
        )

    @property
    def hr_t(self) -> np.ndarray:
        if self._hr_t is None:
            self._lazy_load(PolarDataKind.HR)
        return run_vector_property(
            "hr_t", "hr_t", self._cast, self._hr_t, self._hr_t, self.hr_t_start, self.hr_t_stop, self.resampling
        )

    @property
    def ecg(self) -> np.ndarray:
        if self._ecg is None:
            self._lazy_load(PolarDataKind.ECG)
        return run_vector_property(
            "ecg", "ecg_t", self._cast, self._ecg, self._ecg_t, self.ecg_t_start, self.ecg_t_stop, self.resampling
        )

    @property
    def ecg_t(self) -> np.ndarray:
        if self._ecg_t is None:
            self._lazy_load(PolarDataKind.ECG)
        return run_vector_property(
            "ecg_t", "ecg_t", self._cast, self._ecg_t, self._ecg_t, self.ecg_t_start, self.ecg_t_stop, self.resampling
        )

    @property
    def accel_x(self) -> np.ndarray:
        if self._accel_x is None:
            self._lazy_load(PolarDataKind.ACC)
        return run_vector_property(
            "accel_x", "accel_t", self._cast, self._accel_x, self._accel_t, self.accel_t_start, self.accel_t_stop, self.resampling
        )

    @property
    def accel_y(self) -> np.ndarray:
        if self._accel_y is None:
            self._lazy_load(PolarDataKind.ACC)
        return run_vector_property(
            "accel_y", "accel_t", self._cast, self._accel_y, self._accel_t, self.accel_t_start, self.accel_t_stop, self.resampling
        )

    @property
    def accel_z(self) -> np.ndarray:
        if self._accel_z is None:
            self._lazy_load(PolarDataKind.ACC)
        return run_vector_property(
            "accel_z", "accel_t", self._cast, self._accel_z, self._accel_t, self.accel_t_start, self.accel_t_stop, self.resampling
        )

    @property
    def accel_t(self) -> np.ndarray:
        if self._accel_t is None:
            self._lazy_load(PolarDataKind.ACC)
        return run_vector_property(
            "accel_t", "accel_t", self._cast, self._accel_t, self._accel_t, self.accel_t_start, self.accel_t_stop, self.resampling
        )

    @classmethod
    def from_raw_h10(cls, raw: PolarH10Raw | list[PolarH10Raw]) -> "PolarH10DataSet":
        return cls(raw)

    @classmethod
    def from_file(
        cls,
        file: PathLike,
        serial: str | None = None,
        incl_empty: bool = False,
        auto_recover: bool = True,
    ) -> Mapping[str, "PolarH10DataSet"] | "PolarH10DataSet":
        del auto_recover
        file = Path(file).expanduser()
        if not file.exists():
            return cls.empty()

        try:
            raw = parse_polar_sqlite(file, serial)
        except sqlite3.DatabaseError:
            log.error("Unable to parse sqlite db.", exc_info=True)
            return cls.empty()

        if isinstance(raw, dict):
            result: dict[str, PolarH10DataSet] = {}
            for devname, objs in raw.items():
                keep = [obj for obj in objs if incl_empty or obj.has_data]
                single = cls.from_raw_h10(keep)
                single.sources.append(file)
                result[devname] = single
            return result if result else cls.empty()

        result = cls.from_raw_h10(raw)
        result.sources.append(file)
        return result

    def resample(
        self,
        ecg_t: ResamplingOptions | None = None,
        hr_t: ResamplingOptions | None = None,
        accel_t: ResamplingOptions | None = None,
        **kwargs,
    ) -> None:
        super().resample(ecg_t=ecg_t, hr_t=hr_t, accel_t=accel_t, **kwargs)
