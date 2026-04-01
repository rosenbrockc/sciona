"""Minimal standalone copies of parser helper functions and classes.

This module eliminates the dependency on ``sciona.datasets.parser`` (the
external parser package) by bundling only the exact
functions and classes that ``sciona.principal.datasets`` actually uses.
"""

from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from fnmatch import fnmatch
from importlib import import_module
from os import PathLike, chdir as os_chdir, getcwd, path, stat
from pathlib import Path
from typing import Callable, List, Mapping, Tuple, Union

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vendored utility helpers
# ---------------------------------------------------------------------------

IMPORT_CACHE = {}


def _sort_token(value) -> tuple[int, float | str]:
    if isinstance(value, datetime):
        return (0, value.timestamp())
    if isinstance(value, date):
        return (0, float(value.toordinal()))
    if isinstance(value, (int, float)):
        return (1, float(value))
    return (2, str(value))


def _metadata_start_value(meta) -> object:
    datafile = getattr(meta, "datafile", None)
    datafile_start = getattr(datafile, "start", None)
    if datafile_start is not None:
        return datafile_start
    return meta.time


def _metadata_has_valid_start(meta) -> bool:
    value = _metadata_start_value(meta)
    if isinstance(value, (datetime, date)):
        return True
    if isinstance(value, (int, float)):
        return value > 0
    return value is not None


def _metadata_sort_key(meta) -> tuple[int, float | str]:
    return _sort_token(_metadata_start_value(meta))


@contextmanager
def chdir(target):
    """Context manager for executing some code within a different
    directory after which the current working directory will be set
    back to what it was before.
    Args:
        target (str): path to the directory to change into.
    """
    current = getcwd()
    try:
        os_chdir(target)
        yield target
    finally:
        os_chdir(current)


def import_fqn(fqdn, folder=None, cache: bool = True):
    """Returns the object from the specified fully-qualified name.
    Any exceptions raised will bubble up.

    Args:
        fqdn (str): '.'-separated list of `package.module.callable` to
          import. The callable will *not* be called.
        folder (str): a folder to perform the import inside of.
        cache: when `True`, use the global in-memory cache to optimize
            repeated imports of the same FQNs.

    Returns:
        tuple: `(module, callable)`, where `module` is the module object that
        `callable` resides in.
    """
    global IMPORT_CACHE
    if cache and fqdn in IMPORT_CACHE:
        return IMPORT_CACHE[fqdn]

    if folder is not None:
        with chdir(folder):
            result = import_fqn(fqdn, cache=cache)
        return result

    parts = fqdn.split(".")
    call = parts[-1]
    module = ".".join(parts[0:-1])
    log.debug(f"Importing {module} dynamically.")

    try:
        module = import_module(module)
        if not hasattr(module, call):
            module = import_module(fqdn)
    except (ImportError, ModuleNotFoundError):
        log.debug(f"Import error for {fqdn}. Trying separate import.", exc_info=True)
        try:
            module = import_module(parts[0])
            if len(parts) > 2:
                for part in parts[1:-1]:
                    module = getattr(module, part)
        except (ImportError, ModuleNotFoundError):
            log.debug(f"Import error for {parts[0]}.", exc_info=True)
            module = None

    if module is not None and hasattr(module, call):
        call = getattr(module, call)
        result = module, call
    elif module is not None:
        result = module, None
    else:
        result = None, None

    if cache:
        IMPORT_CACHE[fqdn] = result
    return result


def le_fuzzy(a, b, tol: float = 1e-9):
    """Fuzzy less-than-or-equal comparison of two values."""
    return a < b + tol or abs(a - b) < tol


def ge_fuzzy(a, b, tol: float = 1e-9):
    """Fuzzy greater-than-or-equal comparison of two values."""
    return a > b - tol or abs(a - b) < tol


def overlaps_fuzzy(a, b, tol: float = 1e-9):
    """Determines the region of overlap between two monotonically increasing
    (time) series.
    """
    if len(a) == 0 or len(b) == 0:
        return

    _a, _b = a[0], b[0]
    a_, b_ = a[-1], b[-1]

    if b_ > a_ + tol and _b > a_ + tol:
        return
    if a_ > b_ + tol and _a > b_ + tol:
        return

    if abs(_a - _b) < tol and abs(a_ - b_) < tol:
        return _a, a_

    if le_fuzzy(_a, _b, tol) and le_fuzzy(b_, a_, tol):
        return _b, b_
    if le_fuzzy(_b, _a, tol) and le_fuzzy(a_, b_, tol):
        return _a, a_

    if le_fuzzy(_a, _b, tol) and (ge_fuzzy(b_, a_, tol)):
        return (_b, a_)

    if le_fuzzy(_b, _a, tol) and (ge_fuzzy(a_, b_, tol)):
        return (_a, b_)


def overlaps(a, b, tol: float = 1e-9):
    """Determines the region of overlap between two monotonically increasing
    (time) series.
    """
    o = overlaps_fuzzy(a, b, 1e-9)
    if o is None:
        return

    s, e = o
    if abs(s - e) < tol:
        return None
    return s, e


def execute_any_transform(spec: dict, value, meta: dict = None):
    """Dynamically imports a transform function if it exists in `spec`.
    Adds any `kwargs` definitions, etc.
    """
    transform: dict = spec.get("transform")
    if transform is not None:
        kwargs: dict = transform.get("kwargs", {}).copy()
        if meta is not None:
            if isinstance(meta, dict):
                kwargs.update(meta)
            else:
                kwargs["meta"] = meta

        exclude = transform.get("exclude", [])
        for xname in exclude:
            del kwargs[xname]

        _, caller = import_fqn(transform["fqn"])
        if caller is None:
            raise RuntimeError(f"Unable to import transformer function {transform}.")

        log.debug(
            "Transforming templated data set attribute result using %r.", caller,
        )
        try:
            return caller(value, **kwargs)
        except:
            log.error(
                "Transform function %r failed.", caller, exc_info=True,
            )

    return value


# ---------------------------------------------------------------------------
# Vendored filtering helpers
# ---------------------------------------------------------------------------

DEFAULT_ACC_FS = 26.0
DEFAULT_EDA_FS = 8.0
DEFAULT_TEMP_FS = 0.25
DEFAULT_PPG_FS = 100.0
DEFAULT_AMB_FS = 10.0


class ResamplingMethod(Enum):
    INTERP = 0
    NONE = 1


class ResamplingOptions(object):
    """Data container for the options in resampling a signal."""
    def __init__(self,
            fs: float,
            method: ResamplingMethod = ResamplingMethod.INTERP,
            min_gap: float = 5.0,
            remove_nan: bool = False
    ) -> None:
        self.fs = fs
        self.method = method
        self.min_gap = min_gap
        self.remove_nan = remove_nan

    @staticmethod
    def eda_default():
        return ResamplingOptions(DEFAULT_EDA_FS)

    @staticmethod
    def ppg_default():
        return ResamplingOptions(DEFAULT_PPG_FS)

    @staticmethod
    def accel_default():
        return ResamplingOptions(DEFAULT_ACC_FS)

    @staticmethod
    def ambient_light_default():
        return ResamplingOptions(DEFAULT_AMB_FS)

    @staticmethod
    def skin_temp_default():
        return ResamplingOptions(DEFAULT_TEMP_FS, min_gap=2.5 / DEFAULT_TEMP_FS)

    @staticmethod
    def ambient_temp_default():
        return ResamplingOptions(DEFAULT_TEMP_FS)


def resample_data(
        ts: np.ndarray, data: np.ndarray, fs: int, remove_nan=False,
        min_gap_for_signal=5
    ) -> Tuple[np.ndarray, np.ndarray]:

    min_valid_timestamp = 946684800  # Unix timestamp for 2000-01-01 00:00:00 GMT
    valid_mask = ts > min_valid_timestamp

    ts = ts[valid_mask]

    if len(data.shape) == 1:
        data = data.reshape(data.shape[0], 1)
    data = data[valid_mask]
    col_num = data.shape[1]
    big_gap_ind = np.hstack(
        (np.where(np.diff(ts) > min_gap_for_signal)[0] + 1, len(ts) + 1)
    )
    new_ts, new_vals = [], []
    new_ts.append(ts[: big_gap_ind[0] - 1])
    new_vals.append(data[: big_gap_ind[0] - 1 :])
    for i in range(len(big_gap_ind) - 1):
        new_ts.append(
            ts[big_gap_ind[i] - 1] + (ts[big_gap_ind[i]] - ts[big_gap_ind[i] - 1]) / 2
        )
        new_vals.append([np.nan] * col_num)
        new_ts.append(ts[big_gap_ind[i] : big_gap_ind[i + 1]])
        new_vals.append(data[big_gap_ind[i] : big_gap_ind[i + 1], :])
    new_ts = np.hstack(new_ts)
    new_vals = np.vstack(new_vals)
    time = np.arange(np.nanmin(new_ts), np.nanmax(new_ts), 1 / fs)
    interp_data = np.array(
        [np.interp(time, new_ts, new_vals[:, j]) for j in range(col_num)]
    ).T
    if remove_nan:
        mask = ~np.isnan(interp_data[:, 0])
        time = time[mask]
        interp_data = interp_data[mask, :]
    if col_num == 1:
        interp_data = interp_data.ravel()
    return time, interp_data


def resample(time, signal, options: ResamplingOptions) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a signal to have an alternate sampling rate."""
    if options.method == ResamplingMethod.INTERP:
        if len(time) == 0:
            return time, signal

        return resample_data(
            time, signal, options.fs, options.remove_nan, options.min_gap
        )
    else:
        return time, signal


def filter_times(ts, ts_edges, exclude=True, l_buffer=0, r_buffer=0):
    """Filters timestamps based on edges (including or excluding)."""
    if exclude:
        mask = np.ones(len(ts), dtype=bool)
        for edge in ts_edges:
            mask *= (ts < edge[0] - l_buffer) | (ts >= edge[1] + r_buffer)
    else:
        mask = np.zeros(len(ts), dtype=bool)
        for edge in ts_edges:
            mask = mask | ((ts >= edge[0] - l_buffer) & (ts < edge[1] + r_buffer))
    ts_masked = ts[mask]
    return ts_masked, mask


def find_time_edge(t: np.ndarray, ts: float, left: bool = False) -> int:
    """Finds the first index in the time array `t` that is greater than `ts`."""
    if len(t) == 0:
        raise ValueError("Array length of zero is unacceptable.")

    o = overlaps((t[0], t[-1]), (ts - 1e-6, ts + 1e-6))
    if o is None:
        if ts < t[0] or abs(ts - t[0]) < 1e-6:
            return 0
        else:
            return len(t) - 1

    if ts > t[-1] or abs(ts - t[-1]) < 1e-6:
        return len(t) - 1

    result = np.argmax(t > ts)
    if left:
        return max(0, result - 1)
    return result


# ---------------------------------------------------------------------------
# Vendored base-model helpers
# ---------------------------------------------------------------------------

SUPERFRAME_REVERSE_MAP = {
    "ambient_temp": "temp_amb",
    "skin_temp": "temp_skin",
    "ambient_light": "amb_light",
}


def get_attr_groups(
        time_series: List[str], obj, private: bool = True, exclude: List[str] = None,
        use_reverse_map: bool = True,
    ) -> Mapping[str, List[str]]:
    """Extracts groupings of attributes using the specified list of
    `time_series` attributes as a base.
    """
    if exclude is not None and not isinstance(exclude, (list, tuple)):
        exclude = [exclude]

    groups: Mapping[str, List[str]] = {}
    for tattr in time_series:
        parts = tattr.split('_')
        group = '_'.join(parts[:-1])
        groups[group] = []

    for group in groups:
        name = group
        if use_reverse_map and group in SUPERFRAME_REVERSE_MAP:
            name = SUPERFRAME_REVERSE_MAP[group]

        if private:
            N, prefix = len(name) + 1, f"_{name}"
        else:
            N, prefix = len(name), name

        for attr in dir(obj):
            if attr[0 : N] == prefix:
                if exclude is None or not any(fnmatch(attr, p) for p in exclude):
                    groups[group].append(attr)

    return groups


@dataclass
class FolderFilterOptions:
    """Simple container for filter options from a data collection's ``from_folder``."""
    ext: str
    regex: re.Pattern
    user: str = None
    serial: str = None
    recursive: bool = True
    cro: str = None
    day: date = None
    within_days: int = 1


class UserDay:
    """Placeholder for the UserDay type used in autoload signatures."""
    pass


# ---------------------------------------------------------------------------
# DataSetBase — minimal base class
# ---------------------------------------------------------------------------

class DataSetBase(object):
    """Minimal base class for datasets compatible with the templated pipeline."""

    SUPPORTS_STREAMING: bool = False
    MULTIDATA: bool = False
    TIME_SERIES: List[str] = None

    def __init__(self) -> None:
        self._start = None
        self._stop = None
        self._cast = {}

        if self.TIME_SERIES is not None:
            for series in self.TIME_SERIES:
                setattr(self, f"{series}_start", None)
                setattr(self, f"{series}_stop", None)

        self.sources: List[Path] = []

        self.mask_regions = None
        self.mask_exclude = None
        self.mask_lbuffer = None
        self.mask_rbuffer = None

        self.resampling = {}

    def lazy_load_attrs(self):
        log.debug(f"Skipping lazy load on {self}; this class does not use lazy attributes.")

    @classmethod
    def empty(cls) -> "DataSetBase":
        raise NotImplementedError()

    def __len__(self):
        if self.TIME_SERIES is not None:
            return min([len(getattr(self, t)) for t in self.TIME_SERIES])
        raise NotImplementedError()

    def to_pandas(self,
            private: bool = True, exclude: List[str] = None,
            group_only: Union[str, List[str]] = None, group_exclude: Union[str, List[str]] = None,
        ) -> pd.DataFrame:
        if group_only is not None and not isinstance(group_only, (list, tuple)):
            group_only = [group_only]
        if group_exclude is not None and not isinstance(group_exclude, (list, tuple)):
            group_exclude = [group_exclude]

        groups: Mapping[str, List[str]] = self.get_attr_groups(self, private=private, exclude=exclude)
        result = {}

        for group, attrs in groups.items():
            if group_only is not None and not any(fnmatch(group, o) for o in group_only):
                continue
            if group_exclude is not None and any(fnmatch(group, x) for x in group_exclude):
                continue

            gd = {}
            for attr in attrs:
                key = attr if attr[0] != '_' else attr[1:]
                val = getattr(self, attr)
                if val is not None:
                    gd[key] = val

            result[group] = pd.DataFrame(gd)

        return result

    @classmethod
    def get_attr_groups(cls,
            source: "DataSetBase", private: bool = True, exclude: List[str] = None,
        ) -> Mapping[str, List[str]]:
        return get_attr_groups(cls.TIME_SERIES, source, private=private, exclude=exclude)

    @classmethod
    def merge(cls, *sources: List["DataSetBase"],
              meta: "UserMetaData" = None,
              exclude: List[str] = None,
    ) -> "DataSetBase":
        if len(sources) == 0:
            log.error(f"Cannot merge zero sources in {cls}")
            return
        if any(not isinstance(s, cls) for s in sources):
            log.error("Can't combine sources of different types.")
            return

        groups: Mapping[str, List[str]] = cls.get_attr_groups(sources[0], exclude=exclude)
        data: Mapping[str, Mapping[str, List[np.ndarray]]] = {}
        for group, attrs in groups.items():
            data[group] = {}
            for attr in attrs:
                data[group][attr] = []
                for source in sources:
                    source.lazy_load_attrs()
                    data[group][attr].append(getattr(source, attr))

        result = sources[0].empty()
        for group, d in data.items():
            for attr, arrays in d.items():
                setattr(result, attr, np.concatenate(arrays))

        return result

    @property
    def min(self) -> float:
        if self.TIME_SERIES is not None:
            okay = [
                np.min(getattr(self, t)) for t in self.TIME_SERIES
                if len(getattr(self, t)) > 0
            ]
            if len(okay) > 0:
                return min(okay)
        else:
            raise NotImplementedError()

    @property
    def max(self) -> float:
        if self.TIME_SERIES is not None:
            okay = [
                np.max(getattr(self, t)) for t in self.TIME_SERIES
                if len(getattr(self, t)) > 0
            ]
            if len(okay) > 0:
                return max(okay)
        else:
            raise NotImplementedError()

    def _reslice(self):
        if self.TIME_SERIES is None:
            raise NotImplementedError()

        self._cast = {}

        if self._start is not None:
            for series in self.TIME_SERIES:
                if len(getattr(self, series)) > 0:
                    edge = find_time_edge(getattr(self, series), self._start)
                    setattr(self, f"{series}_start", edge)
        else:
            for series in self.TIME_SERIES:
                setattr(self, f"{series}_start", 0)

        if self._stop is not None:
            for series in self.TIME_SERIES:
                if len(getattr(self, series)) > 0:
                    edge = find_time_edge(getattr(self, series), self._stop)
                    setattr(self, f"{series}_stop", edge)
        else:
            for series in self.TIME_SERIES:
                _t = getattr(self, series)
                setattr(self, f"{series}_stop", len(_t))

        if self.mask_regions is not None:
            for series in self.TIME_SERIES:
                tall = getattr(self, series)
                s, e = getattr(self, f"{series}_start"), getattr(self, f"{series}_stop")
                _t = tall[s : e]
                _, self._cast[f"{series}_mask"] = filter_times(
                    _t, self.mask_regions,
                    exclude=self.mask_exclude, l_buffer=self.mask_lbuffer, r_buffer=self.mask_rbuffer,
                )

    def slice(self, start: float = None, stop: float = None):
        if stop is None or stop > 0:
            self._start = start
            self._stop = stop
        elif len(self.t) > 0:
            self._stop = self.max
            self._start = self._stop + stop

        self._reslice()

    @staticmethod
    def from_file(file: PathLike, serial: str = None) -> "DataSetBase":
        raise NotImplementedError()

    @classmethod
    def from_folder(cls, folder: PathLike, serial: str = None) -> "DataSetBase":
        raise NotImplementedError()

    def mask(self,
            regions: List[Tuple[float, float]] = None,
            lbuffer: float = None,
            rbuffer: float = None
        ):
        self.mask_regions = regions
        self.mask_lbuffer = lbuffer
        self.mask_rbuffer = rbuffer
        self.mask_exclude = False

        self._reslice()

    def resample(self, **kwargs):
        if self.TIME_SERIES is None:
            raise NotImplementedError()

        self.resampling.update({
            k: v for k, v in kwargs.items()
            if k in self.TIME_SERIES and isinstance(v, ResamplingOptions)
        })
        self._reslice()


# ---------------------------------------------------------------------------
# UserMetaData — minimal base class
# ---------------------------------------------------------------------------

class UserMetaData(object):
    """Minimal base class for metadata attached to a single data file."""

    META_EXT: str = ".json"
    DATA_EXT: Union[str, List[str]] = None
    SOURCE = None
    ISOURCE = None
    DATASET: DataSetBase = None

    def __init__(self,
            filepath: PathLike, folder: PathLike = None, datafile=None,
            cache: PathLike = None, metadata: dict = None, connections: list = None,
        ):
        if self.META_EXT is None:
            raise TypeError(f"Sub-class {self} did not override meta data file extension.")
        if self.DATA_EXT is None:
            raise TypeError(f"Sub-class {self} did not override data file extension.")
        if self.SOURCE is None:
            raise TypeError(f"Sub-class {self} did not override any file source sub-class.")
        if self.DATASET is None:
            raise TypeError(f"Sub-class {self} did not override data set sub-class.")

        self.connections: list = connections
        self.dserial: str = None

        if metadata is None:
            self._load_standard(filepath, folder, datafile, cache)
        else:
            self._load_explicit(metadata, filepath, folder, datafile)

        self.filename = self.datafile
        self._sources: List[Path] = None
        self._source_hash: str = None

        DATA_EXTS = self.DATA_EXT
        if not isinstance(DATA_EXTS, (list, tuple)):
            DATA_EXTS = [DATA_EXTS]
        if not any(self.ext == f".{e}" for e in DATA_EXTS):
            if self.ext != self.META_EXT:
                raise ValueError(
                    f"Specified file path {filepath} is not any of {DATA_EXTS} or "
                    f"{self.META_EXT} file types."
                )

        self._start, self._stop = None, None
        self._data = None
        self._offset: float = None
        self._tz_logged: bool = False

    def _load_explicit(self,
            metadata: dict, filepath: PathLike, folder: PathLike, datafile
        ):
        f = Path(filepath)
        if not f.exists():
            raise FileNotFoundError(
                f"The data file at {filepath} does not exist. Data file existence "
                "is required when using explicit metadata objects."
            )

        self.folder = folder
        if self.folder is None:
            self.folder = path.dirname(filepath)
        self.cache = self.folder
        self.datafile = datafile
        self.filepath = Path(filepath)
        self.name, self.ext = path.splitext(path.basename(self.filepath))
        self.metafile = Path(path.join(self.folder, f"explicit{self.META_EXT}"))
        self._metadata = metadata
        self.metasize = len(json.dumps(metadata))

    def _load_standard(self,
            filepath: PathLike, folder: PathLike, datafile, cache: PathLike
        ):
        if folder is not None:
            self.folder = folder
        else:
            self.folder = path.dirname(filepath)
            if self.folder == "":
                self.folder = getcwd()

        self.cache = Path(cache if cache is not None else self.folder)

        name, ext = path.splitext(path.basename(filepath))
        if ext == self.META_EXT:
            filepath = Path(path.join(self.folder, f"{name}.{self.DATA_EXT}"))
            if not filepath.exists():
                filepath = Path(path.join(self.folder, name))

        self.filepath = Path(filepath)
        self.name, self.ext = path.splitext(path.basename(self.filepath))
        self.metafile = Path(path.join(self.folder, f"{self.name}{self.META_EXT}"))
        if not self.metafile.exists():
            self.metafile = Path(path.join(self.folder, f"{self.name}{self.ext}{self.META_EXT}"))

        self._metadata = None
        self.metasize = 0

        self.datafile = datafile
        if datafile is None and self.SOURCE is not None:
            target = str(self.filepath).replace(str(self.cache), "")
            if target[0] == '/':
                target = target[1:]
            self.datafile = self.SOURCE.parse(target)

    def load_meta_file(self):
        if self.META_EXT != ".json":
            raise TypeError(
                f"The sub-class uses a non-JSON metadata file type {self.META_EXT}, "
                "but has not overloaded :meth:`load_meta_file`."
            )

        if path.isfile(self.metafile):
            log.debug(f"Loading inferred metadata from {self.metafile}.")
            with open(self.metafile) as f:
                self._metadata = json.load(f)

            self.metasize = stat(self.metafile).st_size
        else:
            log.debug(f"Inferred metadata file {self.metafile} does not exist.")

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            self.load_meta_file()
        return self._metadata

    @property
    def time(self):
        return self.start

    @property
    def start(self):
        self._get_start_stop()
        return self._start

    @property
    def stop(self):
        self._get_start_stop()
        return self._stop

    def _get_start_stop(self):
        if self._start is None:
            if self._data is not None and len(self._data) > 0:
                self._start, self._stop = self._data.min, self._data.max
            else:
                self._start, self._stop = -1.0, -1.0

    @property
    def user(self):
        return self.filename.user

    @property
    def data(self) -> DataSetBase:
        if self._data is None:
            if self.DATASET.MULTIDATA:
                self._data = self.DATASET.from_folder(self.folder, meta=self)
            else:
                if not path.isfile(self.filepath):
                    log.error(f"{self.filepath} does not exist; can't load file.")
                    return
                self._data = self.DATASET.from_file(self.filepath)

            if isinstance(self._data, dict):
                if len(self._data) == 0:
                    self._data = self.DATASET.empty()
                elif len(self._data) == 1:
                    self._data = next(iter(self._data.values()))
                elif self.dserial is not None and self.dserial in self._data:
                    self._data = self._data[self.dserial]
                elif self.datafile.serial in self._data:
                    self._data = self._data[self.datafile.serial]

            if self._data is not None:
                self._sources = self._data.sources

        return self._data

    @property
    def device(self):
        raise NotImplementedError()

    @property
    def user_id(self) -> str:
        raise NotImplementedError()

    @property
    def is_valid(self) -> bool:
        raise NotImplementedError()


# ---------------------------------------------------------------------------
# DataSetCollection — minimal base class
# ---------------------------------------------------------------------------

class DataSetCollection(object):
    """Minimal base class for a collection of dataset files for a single user."""

    DATA_CLASS: DataSetBase = None
    META_CLASS: UserMetaData = None
    CUSTOM_PARSER: Callable = None

    def __init__(self, user_id: str, files: List[UserMetaData], label: str = None):
        if self.META_CLASS is None:
            raise TypeError(f"Sub-class {self} did not override the metadata class.")
        if self.DATA_CLASS is None:
            raise TypeError(f"Sub-class {self} did not override the individual data class definition attribute.")

        self.user_id = user_id
        self.files = files
        self.label = label

        self._valid: List[UserMetaData] = None
        self._filtered: List[UserMetaData] = None
        self._ordered: List[UserMetaData] = None

        self._start, self._stop = None, None
        self.data: DataSetBase = None
        self.static_cls = None
        self.resampling = {}
        self.mask_regions = None
        self.mask_lbuffer = None
        self.mask_rbuffer = None

        self.transformer = None

    @property
    def has_data(self) -> bool:
        return len(self.ordered) > 0

    @property
    def valid(self) -> List[UserMetaData]:
        if self._valid is None:
            self._valid = [f for f in self.files if _metadata_has_valid_start(f)]
        return self._valid

    @property
    def filtered(self) -> List[UserMetaData]:
        if self._filtered is None:
            self._filtered = self.valid.copy()
        return self._filtered

    @property
    def ordered(self) -> List[UserMetaData]:
        if self._ordered is None:
            self._ordered = sorted(self.filtered, key=_metadata_sort_key)
        return self._ordered

    @property
    def meta(self) -> UserMetaData:
        if len(self.files) == 0:
            return
        return self.files[0]

    def get_meta(self, uday) -> UserMetaData:
        for meta in self.files:
            d = meta.datafile
            if hasattr(uday, 'user') and hasattr(uday, 'serial') and hasattr(uday, 'day'):
                if d.user == uday.user and d.serial == uday.serial and d.day == uday.day:
                    return meta
        return self.meta

    @classmethod
    def get_filter_options(cls,
            user: str, serial: str, recursive: bool = True, day: date = None,
        ) -> FolderFilterOptions:
        matcher = cls.META_CLASS.SOURCE.matcher()
        ext = cls.META_CLASS.DATA_EXT
        return FolderFilterOptions(ext, matcher, user, serial, recursive, day=day)

    def load_all(self,
            loader: Callable[[UserMetaData], DataSetBase] = None,
            merger: Callable[[List[DataSetBase]], DataSetBase] = None,
        ):
        self.data = None
        contents: List[DataSetBase] = []
        for meta in self.ordered:
            if loader is not None:
                data = loader(meta)
            else:
                data = meta.data

            if self.transformer is not None:
                data = self.transformer(data)

            contents.append(data)

        if len(contents) == 0:
            self.data = None
            log.info("No metadata files to generate static class from. Skipping.")
            return

        if merger is not None:
            self.data = merger(*contents)
        else:
            self.data = type(contents[0]).merge(*contents, meta=self.meta)

        if self.data is not None:
            self.data.slice(self._start, self._stop)
            self.data.mask(self.mask_regions, self.mask_lbuffer, self.mask_rbuffer)
            self.data.resample(**self.resampling)
        else:
            log.debug(f"Skipping slice, mask and resample; data is None. From {self.ordered}.")

    def to_pandas(self,
            loader: Callable[[UserMetaData], DataSetBase] = None,
            merger: Callable[[List[DataSetBase]], DataSetBase] = None,
            exclude: List[str] = None,
        ) -> Mapping[str, pd.DataFrame]:
        if self.data is None:
            self.load_all(loader, merger)
            if self.data is None:
                return {}
        return self.data.to_pandas(exclude=exclude)

    def slice(self, start: float = None, stop: float = None):
        if stop is None or stop > 0:
            self._start = start
            self._stop = stop
        elif len(self.ordered) > 0:
            self._stop = self.ordered[-1].stop
            self._start = self._stop + stop

        if self.data is not None:
            self.data.slice(self._start, self._stop)

    def mask(self,
            regions: List[Tuple[float, float]] = None,
            lbuffer: float = None,
            rbuffer: float = None
        ):
        self.mask_regions = regions
        self.mask_lbuffer = lbuffer
        self.mask_rbuffer = rbuffer
        if self.data is not None:
            self.data.mask(regions, lbuffer, rbuffer)

    def __getitem__(self, index):
        return self.ordered[index]

    def __len__(self):
        return len(self.valid)

    @classmethod
    def from_folder(cls,
            folder: Path, options: FolderFilterOptions = None,
            subcls: "DataSetCollection" = None, dserial: str = None,
            label: str = None,
        ) -> Union["DataSetCollection", Mapping]:
        if subcls is None:
            subcls = cls
        if options is None:
            options = subcls.get_filter_options(None, None)

        regex = options.regex
        if options.regex is None:
            regex = subcls.META_CLASS.SOURCE.matcher()
        if options.ext is None:
            options.ext = subcls.META_CLASS.DATA_EXT

        if subcls.CUSTOM_PARSER is not None:
            files = subcls.CUSTOM_PARSER(folder, regex, options.ext, subcls, options.recursive)
        else:
            raise NotImplementedError(
                "Standard parse_folder is not available in the standalone module. "
                "Only CUSTOM_PARSER-based dataset collections are supported."
            )

        if options.user == '*':
            result, metaset = {}, {}
            for metas in files.values():
                for meta in metas:
                    uday_key = (meta.datafile.user, meta.datafile.serial, meta.datafile.start)
                    if uday_key not in metaset:
                        metaset[uday_key] = []
                    metaset[uday_key].append(meta)

            for uday_key, metafiles in metaset.items():
                result[uday_key] = subcls(uday_key[0], metafiles, label=label)

            return result

        user = options.user
        if options.user is None:
            users = sorted(files.keys())
            user = users[0]

        if user not in files:
            log.debug(
                "The user %s does not have any files to load in %s. "
                "Assuming lazy evaluation and returning empty data set.",
                user, folder,
            )
            filts = []
        else:
            filts: List[UserMetaData] = files[user]

        if options.serial is not None:
            filts = [
                meta for meta in filts if meta.datafile.serial == options.serial
            ]

        if options.day is not None:
            filts = [
                meta for meta in filts
                if abs((meta.datafile.day - options.day).days) <= options.within_days
            ]

        if dserial is not None:
            for meta in filts:
                meta.dserial = dserial

        return subcls(options.user, filts, label=label)
