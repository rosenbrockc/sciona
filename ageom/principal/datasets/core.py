"""Core templated dataset classes and helpers.

Extracted from ``ageom.datasets.templated``, retaining only the generic
adapter-driven dataset machinery.
"""

import numpy as np
from pathlib import Path
from os import PathLike, path
import pandas as pd
import logging
from typing import Mapping, List, Tuple, Union, Callable
import re
from operator import itemgetter
import json
from hashlib import sha256
from dataclasses import dataclass
from datetime import datetime, date

from ageom.datasets.parser.base import (
    DataSetBase,
    UserMetaData,
    DataSetCollection,
    FolderFilterOptions,
    get_attr_groups,
)


@dataclass
class DataFileName:
    """Data container representing a data file from a data source.

    Attributes:
        user: the user-specific identifier.
        serial: the serial number/identifier of the hardware device.
        start: parsed date/time that the file was created or uploaded.
        stop: for file names that represent a range in time, the stop
            time. Otherwise ``None``.
    """
    user: str
    serial: str
    start: datetime
    stop: datetime = None

    @property
    def day(self):
        """Returns the *date* part of :attr:`start`."""
        if isinstance(self.start, datetime):
            return self.start.date()
        return self.start
from ageom.datasets.parser.utils import import_fqn, execute_any_transform
from ageom.datasets.parser.filtering import (
    ResamplingOptions, ResamplingMethod, filter_times, resample, find_time_edge,
)

from .io import read

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging shim (replaces ageom.datasets.utils.msg)
# ---------------------------------------------------------------------------

class _Msg:
    def warn(self, text):
        log.warning(text)

    def err(self, text):
        log.error(text)

msg = _Msg()


# ---------------------------------------------------------------------------
# Module-level caches and constants
# ---------------------------------------------------------------------------

METACLASSES = {}
"""dict: keys are template identifiers/hashes; values are the corresponding
sub-classes of :class:`UserMetaData` created at runtime.
"""
COLLECTION_CLASSES = {}
"""dict: keys are template identifiers/hashes; values are the corresponding
sub-classes of :class:`DataSetCollection` created at runtime.
"""
STD_PROPERTIES = [
    "tz",
]
"""list: of standard metadata property names that the :class:`UserMetaData`
expects to be in an explicit metadata `dict`, and therefore has already wired
up the properties.
"""
UDAY_CACHE = {}
"""dict: cache of the data set collections referenced by user days.
"""
DEFAULT_UDAY_LABEL = "{user}/{serial}/{day}"
"""str: default label to use for udays if the template doesn't specify one.
"""
PER_PARSER_CACHE: Mapping[Path, Mapping[str, List[UserMetaData]]] = {}
"""dict: keys are the parent folders when adapters are running in `per=True`
mode for metadata loading; values are a mapping from user name to the
list of metadata files available for that user.
"""
PARSER_CACHE: Mapping[Path, Mapping[str, List[UserMetaData]]] = {}
"""dict: keys are the file paths to tracker files when adapters are running
in `per=False` mode for metadata loading; values are a mapping from user
name to the rows of the tracker file converted to metadata objects.
"""
MULTI_WARN: Mapping[Path, bool] = {}
"""Keys are adapter file paths; values are booleans indicating whether
the warning has already been issued about merging multiple files matching
a single source pattern.
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def run_vector_property(
        attr: str, tattr: str,
        cast: dict, source: np.ndarray, time: np.ndarray,
        start: int, stop: int,
        resampling: dict,
    ):
    """Runs the standard masking/resampling/slicing logic for lazy
    evaluation of a property in a data set.

    .. warning: `cast` gets mutated by this function call.

    Args:
        attr: name of the sensor *samples* being accessed as a property.
        tattr: name of the time series property whose timestamps correspond
            to the samples in the `attr` vector.
        cast: mapping of data set attribute names to sliced, masked and
            resampled vectors.
        source: array of *sample* values.
        time: array of *time* values corresponding to the elements of in `source`.
        start: starting index for this time series (slicing).
        stop: stopping index for this time series (slicing).
        resampling: mapping of time attributes (aka `tattr`) to :class:`ResamplingOptions`.
    """
    if attr not in cast:
        if source is not None:
            _s, _e = start, stop
            cast[attr] = source[_s:_e]

            if f"{tattr}_mask" in cast:
                cast[attr] = cast[attr][
                    cast[f"{tattr}_mask"]
                ]

            if resampling.get(tattr) is not None:
                if attr == tattr:
                    cast[f"{attr}_orig"] = cast[attr]
                else:
                    tlen = len(time)

                _, cast[attr] = resample(
                    cast[f"{tattr}_orig"],
                    cast[attr],
                    resampling[tattr]
                )
                if attr != tattr:
                    assert len(cast[attr]) == tlen
        else:
            cast[attr] = np.array([])

    return cast[attr]


def merge_templated_group(
        group,
        result: "TemplatedDataSet",
        sources: List["TemplatedDataSet"],
        folder: PathLike = None
    ):
    """Merges all the data for a single sensor group.

    Args:
        group: private attribute of the sensor group in a :class:`TemplatedDataSet`
            sub-class to merge.
        result: the dataset whose attributes should be overwritten with the merged
            data for each property in the group.
        sources: iterable of :class:`TemplatedDataSet` sub-class instances
            to merge the `sensor` for.
        folder: if the data hasn't been loaded yet, the folder to load the data from.
    """
    _order = []
    for i, s in enumerate(sources):
        s.load_group_data(folder=folder)
        tattr = s.rtimes[group]
        t = getattr(s, tattr)
        if len(t) > 0:
            _order.append((t[0], i))

    if len(_order) == 0:
        return

    _order = sorted(_order, key=itemgetter(0))
    combined = {}
    for _, idx in _order:
        dset = sources[idx]
        gspec = dset.groups[group]

        for prop in gspec["properties"]:
            propname = get_prop_name(group, prop)
            value = getattr(dset, f"_{propname}")

            if prop not in combined:
                combined[prop] = []
            combined[prop].append(value)

    for prop, arrays in combined.items():
        propname = get_prop_name(group, prop)
        setattr(result, f"_{propname}", np.concatenate(arrays))


def time_merge_dataframes(dfs: List[pd.DataFrame], t: str) -> pd.DataFrame:
    """Merges multiple dataframes into a single one making sure that they are
    ordered correctly in time.

    Args:
        t: name of the column in `dfs` to use when ordering time.
    """
    order = sorted([(df, df[t].iloc[0]) for df in dfs if len(df) > 0],
                   key=itemgetter(1))
    result = pd.concat([e[0] for e in order])
    result.index = np.arange(len(result))

    return result


def get_source_hash(group: dict) -> str:
    """Gets a hash of the source file pattern and its reader
    with optional `kwargs`, etc.

    Args:
        group: the group specification that includes a `source`
            and `reader` keys.
    """
    d = {k: v for k, v in group.items() if k in ["source", "reader"]}
    s = json.dumps(d)
    h = sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def get_prop_name(group: str, prop: str, private: bool = False) -> str:
    """Gets the name of the templated property accessor based on the custom
    naming convention.
    """
    if prop == "[]":
        propname = group
    else:
        propname = f"{group}_{prop}"

    if private:
        return f"_{propname}"
    return propname


def get_datafile_exts(template: dict, first: bool = True) -> Union[str, List[str]]:
    """Gets the first file extension from any of the group sources in a template
    specification.

    Args:
        template: specification for all groups and metadata.
        first: when `True` only return the first extension found in the templates.
    """
    if "groups" not in template:
        raise KeyError("The template does not contain any groups.")
    if first:
        group = next(iter(template["groups"].values()))
        if "source" not in group:
            raise KeyError(f"The group {group} does not have a source.")

        # data extension does not include the `.`
        return path.splitext(group["source"])[1][1:]
    else:
        result = []
        for group in template["groups"].values():
            if "source" not in group:
                raise KeyError(f"The group {group} does not have a source.")
            result.append(path.splitext(group["source"])[1][1:])

        return result


def _get_datafile_day(spec: dict, data: dict) -> datetime:
    """Gets the date part of the :class:`DataFileName` for a templated user metadata.
    """
    if "source" not in spec["day"] and "transform" not in spec["day"]:
        raise ValueError(f"You must specify either `source` or `transform` for metadata 'day'.")

    source = spec["day"].get("source", "")
    if source not in data and "transform" not in spec["day"]:
        raise ValueError(f"The metadata entry {source} for `day` is missing in metadata {data}.")

    if "transform" in spec["day"]:
        return execute_any_transform(spec["day"], data.get(source, ""), spec)
    else:
        ftime = spec["day"].get("ftime", "%m/%d/%y")
        value = data[source]
        try:
            return datetime.strptime(value, ftime).date()
        except ValueError:
            msg.err(f"Date {value} doesn't match {ftime} pattern; in metadata {data}.")


def _get_datafile_any(spec: dict, data: dict, attr: str):
    """Gets an arbitrary attribute from the :class:`DataFileName` specification.
    """
    if "source" not in spec[attr] and "transform" not in spec[attr]:
        raise ValueError(f"You must specify either `source` or `transform` for metadata '{attr}'.")

    source = spec[attr].get("source", "")
    if source not in data and "transform" not in spec[attr]:
        raise ValueError(f"The metadata entry {source} for `{attr}` is missing in metadata {data}.")

    if "transform" in spec[attr]:
        return execute_any_transform(spec[attr], data.get(source, ""), spec)
    else:
        return data[source]


def datafile_from_spec(spec: dict, data: dict) -> DataFileName:
    """Extracts a :class:`DataFileName` using the template specification
    for metadata.

    Args:
        spec: the `meta` specification section from the template.
        data: the actual metadata object that can either be a single row
            from a group file, or the deserialized contents of a JSON or
            similar file at the trial/individual folder level.
    """
    day = _get_datafile_day(spec, data)
    serial = _get_datafile_any(spec, data, "serial")
    user = _get_datafile_any(spec, data, "user")

    return DataFileName(user, serial, day)


def make_uday_label(lbl_fmt: str, datafile: DataFileName, metadata: dict) -> str:
    """Attempts to make a uday label using the components that :class:`UserMetaData`
    would ordinarily use, for when we don't have a metadata instance.
    """
    ml = metadata.copy()
    for k in ("user", "serial", "day"):
        if k in ml:
            del ml[k]

    return lbl_fmt.format(
        user=datafile.user, serial=datafile.serial, day=datafile.start,
        **ml,
    )


def get_templated_subfolders(folder: Path, regex: re.Pattern, recursive: bool = False):
    """Gets any subfolders in `folder` that match the regex (if it is given).

    Args:
        folder: path to the parent folder to get sub-folders in.
        regex: if specified, the folder path *must* match this regex; otherwise,
            just include all sub-folders.
        recursive: when `True`, run the same function again on any children
            that are selected.
    """
    targets = []
    for child in folder.iterdir():
        if child.is_dir():
            if regex is None:
                targets.append(child)
                if recursive:
                    targets.extend(get_templated_subfolders(child, regex, recursive))
            elif regex.match(child.name) is not None:
                targets.append(child)
                if recursive:
                    targets.extend(get_templated_subfolders(child, regex, recursive))

    return targets


class ZeroSourceFilesError(ValueError):
    """Raised when a templated dataset collection is asked to parse a folder
    but the template does not specify any source files.
    """
    def __init__(self, folder: Path):
        super().__init__(f"The template does not specify any source files; cannot parse {folder}.")
        self.folder = folder


# ---------------------------------------------------------------------------
# TemplatedDataSet
# ---------------------------------------------------------------------------

class TemplatedDataSet(DataSetBase):
    """Single user day dataset for an arbitrary template-specified folder of data.

    Args:
        varset: if any template values should have variable values substituted, specify
            the key-value mappings here.

    Attributes:
        groups: mapping from group name to specification for that group.
        datafiles: mapping from group name to the :class:`Path` where
            the group's data was loaded from.
        data: mapping from the group name the data frame with the group's
            loaded data.
        times: mapping from the group name to the property within that group
            that represents time.
        rtimes: mapping from the group name to the property within that group
            that represents the *raw* time before any transformations.
        starts: mapping from the group name to the class attribute that holds
            the starting index of the slice for the group.
        stops: mapping from the group name to the class attribute that holds
            the ending index of the slice for the group.
        raw_dfs: the merged raw dataframes obtained from one or more files
            matching the `source` pattern and produced by the reader with its
            optional kwargs. These are keyed by the SHA256 hash of the source
            and reader specification `dict` together.
    """
    ADAPTER_FILENAME = "adapter.yml"
    """str: name of the file that this data set uses to define its template.
    """
    MULTIDATA = True


    def __init__(self, template: PathLike, varset: dict = None) -> None:
        super().__init__()
        self.varset = varset
        self.template_file = Path(template).expanduser()
        self.template = read(
            self.template_file.parent, path.splitext(self.template_file.name)[0], varset=varset,
        )

        if "groups" not in self.template:
            raise TypeError(f"Dataset template does not define any groups at {self.template_file}.")

        self.groups: Mapping[str, dict] = self.template["groups"]
        self.datafiles: Mapping[str, Path] = {}
        self.raw_dfs: Mapping[str, pd.DataFrame] = {}
        self.raw_files: Mapping[str, Path] = {}
        self.data: Mapping[str, pd.DataFrame] = {}
        self.times: Mapping[str, str] = {}
        self.rtimes: Mapping[str, str] = {}
        self.starts: Mapping[str, str] = {}
        self.stops: Mapping[str, str] = {}

        self._start = None
        self._stop = None
        self._cast = {}

        self.mask_regions = None
        self.mask_exclude = None
        self.mask_lbuffer = None
        self.mask_rbuffer = None

        self.resampling = {}


    def __len__(self):
        """Gets the length of the shortest time series vector in this dataset.

        .. note:: This is the length *including* any slices or masks. Use :meth:`slice`
            without any parameters to reset the slicing and get the minimum length of
            the underlying data.
        """
        result = None
        for group in self.groups:
            l = len(getattr(self, self.times[group]))
            if result is None:
                result = l
            else:
                result = max(result, l)

        return result


    @property
    def folder(self) -> Union[Path, List[Path]]:
        """Gets the folder where the data files are located.

        .. note:: This is not necessarily the folder where the adapter file
            is located because tracker files can specify many user days.
        """
        paths, names = [], set()
        for p in self.raw_files.values():
            if isinstance(p, list):
                p = p[0]  # Use the first file if multiple files are present.
            name = p.parent.name
            if name not in names:
                names.add(name)
                paths.append(p.parent)

        if len(paths) == 0:
            return None
        elif len(paths) == 1:
            return paths.pop()
        else:
            return list(paths)


    def load(self, name: str, folder: PathLike, meta: UserMetaData = None) -> pd.DataFrame:
        """Loads the dataframe for the specified group specification.

        Args:
            name: name of the group being loaded.
            folder: path to the folder where the data for this dataset and
                group is located.
            meta: user metadata to use in time localization when loading data. If not
                provided, some of the time series may not be localized correctly.
        """
        group = self.groups[name]
        if "reader" not in group:
            raise TypeError(f"A reader configuration was not specified for the {name} group.")
        if "source" not in group:
            raise TypeError(f"A source data file pattern was not specified for the {name} group.")

        rawhash = get_source_hash(group)
        if rawhash in self.raw_dfs:
            log.info(f"Using cached dataframes for group {name} sources.")
            result = self.raw_dfs[rawhash]
            if "transform" in group:
                result = execute_any_transform(group, self.raw_dfs[rawhash], meta=meta)

            if name not in self.datafiles:
                self.datafiles[name] = self.raw_files[rawhash]
            return result

        reader, kwargs = group["reader"]["fqn"], group["reader"].get("kwargs", {})
        if "meta" in kwargs:
            kwargs["meta"] = meta
        _, caller = import_fqn(reader)

        matches = list(Path(folder).expanduser().rglob(group["source"]))
        if len(matches) == 0:
            msg.warn(f"Group `{name}` has no data file matching {group['source']} in {folder}.")
            log.debug(f"Skipping folder {folder}; no file matching {group['source']} pattern.")
            return pd.DataFrame([])

        if len(matches) > 1:
            if self.template_file not in MULTI_WARN:
                MULTI_WARN[self.template_file] = False

            if not MULTI_WARN[self.template_file]:
                log.warning(
                    "Group `%s` has more than one data file matching %s in %s.\n"
                    "Matches: %r. Auto-merging.",
                    name, group['source'], folder,
                    ', '.join([m.name for m in matches]),
                )
                MULTI_WARN[self.template_file] = True

            self.datafiles[name] = matches
            self.raw_files[rawhash] = matches
        else:
            self.datafiles[name] = matches[0]
            self.raw_files[rawhash] = matches[0]

        self.sources.extend(matches)

        item = group["reader"].get("item")
        dfs = [caller(m, **kwargs) for m in matches]
        if item is not None:
            dfs = [df[item] for df in dfs if item in df]

        tdfs = []
        for df in dfs:
            if "transform" in group:
                tdfs.append(execute_any_transform(group, df, meta=meta))
            else:
                tdfs.append(df)

        post_cache = group["reader"].get("cache", "post") == "post"
        target = tdfs if post_cache else dfs
        if len(target) == 1:
            raw = target[0]
        else:
            tcol = group["properties"][group["time"]]["source"]
            raw = time_merge_dataframes(target, tcol)

        self.raw_dfs[rawhash] = raw

        if post_cache:
            result = raw
        else:
            if len(tdfs) == 1:
                result = tdfs[0]
            else:
                tcol = group["properties"][group["time"]]["source"]
                result = time_merge_dataframes(tdfs, tcol)

        return result


    def create_property(self, group: str, prop: str):
        """Creates the property dynamically to access the sliced, masked
        and possibly resampled version of the data represented by `spec`
        within the `group`.

        Args:
            group: name of the property group that `prop` belongs to.
            prop: name of the property within `group` to create.
        """
        spec = self.groups[group]["properties"][prop]
        propname = get_prop_name(group, prop)
        setattr(self, f"_{propname}", None)

        def accessor(self):
            f"""{spec['description']}
            """
            return run_vector_property(
                propname, self.times[group],
                self._cast, getattr(self, f"_{propname}", None), getattr(self, self.rtimes[group]),
                getattr(self, self.starts[group]), getattr(self, self.stops[group]),
                self.resampling,
            )

        setattr(type(self), propname, property(accessor))


    def create_group(self, group: str):
        """Creates the group and its properties.

        Args:
            group: name of the group to create.
        """
        spec = self.groups[group]
        if "properties" not in spec:
            raise TypeError(f"The template for group {group} does not define properties.")
        if "time" not in spec:
            raise TypeError(f"The template for group {group} does not specify the attribute for `time`.")

        self.times[group] = f"{group}_{spec['time']}"
        self.rtimes[group] = f"_{self.times[group]}"
        self.starts[group] = f"{group}_start"
        self.stops[group] = f"{group}_stop"
        for prop in spec["properties"]:
            self.create_property(group, prop)

        setattr(self, self.starts[group], None)
        setattr(self, self.stops[group], None)


    @property
    def duration(self):
        """Returns the maximum duration in time of this templated set when *all*
        data streams are taken into account.
        """
        x, m = self.max, self.min
        if x is not None and m is not None:
            return x - m
        else:
            return 0.0


    @property
    def min(self) -> float:
        """Gets the minimum timestamp from the entire set of time series
        within this data set.
        """
        if len(self.times) == 0:
            return
        tmins = [
            (max(self._start, min(getattr(self, self.rtimes[group])))
             if self._start is not None
             else min(getattr(self, self.rtimes[group]))
            )
            for group in self.groups
            if (getattr(self, self.rtimes[group]) is not None and
                len(getattr(self, self.rtimes[group])) > 0)
        ]
        if len(tmins) > 0:
            return min(tmins)


    @property
    def max(self) -> float:
        """Gets the maximum timestamp from the entire set of time series
        within this data set.
        """
        if len(self.times) == 0:
            return
        tmaxs = [
            (min(self._stop, max(getattr(self, self.rtimes[group])))
             if self._stop is not None
             else max(getattr(self, self.rtimes[group]))
            )
            for group in self.groups
            if (getattr(self, self.rtimes[group]) is not None and
                len(getattr(self, self.rtimes[group])) > 0)
        ]
        if len(tmaxs) > 0:
            return max(tmaxs)


    def _reslice(self):
        self._cast = {}

        if self._start is not None:
            for group in self.groups:
                gt = getattr(self, self.rtimes[group])
                if len(gt) > 0:
                    setattr(
                        self, self.starts[group], find_time_edge(gt, self._start)
                    )
                else:
                    setattr(self, self.starts[group], 0)
        else:
            for group in self.groups:
                setattr(self, self.starts[group], 0)

        if self._stop is not None:
            for group in self.groups:
                gt = getattr(self, self.rtimes[group])
                if len(gt) > 0:
                    setattr(
                        self, self.stops[group], find_time_edge(gt, self._stop)
                    )
                else:
                    setattr(self, self.stops[group], 0)
        else:
            for group in self.groups:
                setattr(self, self.stops[group], len(getattr(self, self.rtimes[group])))

        if self.mask_regions is not None:
            for group in self.groups:
                _t = getattr(self, self.rtimes[group])
                if len(_t) > 0:
                    _t = _t[
                        getattr(self, self.starts[group]) : getattr(self, self.stops[group])
                    ]

                _, self._cast[f"{self.times[group]}_mask"] = filter_times(
                    _t, self.mask_regions,
                    exclude=self.mask_exclude,
                    l_buffer=self.mask_lbuffer,
                    r_buffer=self.mask_rbuffer,
                )


    @classmethod
    def merge(cls, *sources: List["TemplatedDataSet"],
            meta: UserMetaData, srcmeta: List[UserMetaData],
            folder: PathLike = None,
        ):
        """Merges multiple data sets ensuring time is sorted.

        Args:
            srcmeta: the metadata used to create/load each of the `sources`.
            folder: if the data hasn't been loaded yet for the sources; the folder
                to look for data in.
        """
        if len(sources) == 0:
            raise ValueError("Cannot merge zero sources.")
        if any(not isinstance(s, TemplatedDataSet) for s in sources):
            raise TypeError("Can't combine sources of different types.")

        data_cls: TemplatedDataSet = type(sources[0])
        result = data_cls.from_folder(
            sources[0].template_file.parent, adapter=meta.ADAPTER_FILEPATH, meta=None,
        )
        unmerged: TemplatedDataSet = np.where([meta is m for m in srcmeta])[0]
        if len(unmerged) > 0:
            unmerged = sources[unmerged[0]]
        else:
            unmerged = None

        anymerge = False
        for group in sources[0].groups:
            if result.groups[group].get("merge", True):
                merge_templated_group(group, result, sources, folder=folder)
                anymerge = True
            else:
                unmerged.load_group_data(folder=folder)
                gspec = result.groups[group]
                for prop in gspec["properties"]:
                    propname = get_prop_name(group, prop)
                    value = getattr(unmerged, f"_{propname}")
                    setattr(result, f"_{propname}", value)

        if anymerge:
            for key in sources[0].raw_files:
                result.raw_files[key] = []
                result.raw_dfs[key] = []
                for source in sources:
                    if key in source.raw_files:
                        result.raw_files[key].append(source.raw_files[key])
                        result.raw_dfs[key].append(source.raw_dfs[key])

            for group in sources[0].datafiles:
                result.datafiles[group] = []
                for source in sources:
                    if group in source.datafiles:
                        result.datafiles[group].append(source.datafiles[group])
        else:
            result.raw_files = unmerged.raw_files.copy()
            result.raw_dfs = unmerged.raw_dfs.copy()
            result.datafiles = unmerged.datafiles.copy()

        return result


    def load_group_data(self,
            folder: PathLike = None, overwrite: bool = False, meta: UserMetaData = None,
        ):
        """Does the actual reading of data from source files for each group
        in this dataset.

        Args:
            folder: specify the folder to look inside for the data.
            overwrite: when `True`, reload the data in this instance, even if it
                already has data loaded.
            meta: user metadata to use in time localization when loading data.
        """
        if len(self.data) > 0 and not overwrite:
            log.debug("Skipping data load; overwrite=False and we already have data.")
            return

        if folder is None:
            folder = self.template_file.parent
        for group in self.groups:
            self.data[group] = self.load(group, folder, meta=meta)
            gprops = self.groups[group]["properties"]
            for prop in gprops:
                spec = gprops[prop]
                propname = get_prop_name(group, prop, private=True)

                if spec["source"] in self.data[group]:
                    value = self.data[group][spec["source"]].to_numpy()
                else:
                    log.warning(
                        "The source %s was not found in the data for group %s. "
                        "Property %s will be empty.",
                        spec["source"], group, propname,
                    )
                    value = np.array([])

                if "transform" in spec and group in self.datafiles:
                    if isinstance(self.datafiles[group], list):
                        value = execute_any_transform(spec, value, {
                            "filename": [f.name for f in self.datafiles[group]],
                            "meta": meta,
                        })
                    else:
                        value = execute_any_transform(spec, value, {
                            "filename": self.datafiles[group].name,
                            "meta": meta,
                        })
                setattr(self, propname, value)


    def slice(self, start: float = None, stop: float = None):
        """Set the limits in time to return data for.

        .. note:: If `stop < 0`, the set will return the
            *last* `stop` seconds of data (approximately).
        """
        if stop is None or stop > 0:
            self._start = start
            self._stop = stop
        elif len(self.t) > 0:
            self._stop = self.max
            self._start = self._stop + stop

        self._reslice()


    @classmethod
    def find_adapter(cls, folder: Path) -> Path:
        """Recursively searches the parent folders of `folder` until
        a file with name :attr:`ADAPTER_FILENAME` is found.
        """
        adapter = folder.joinpath(cls.ADAPTER_FILENAME)
        if adapter.exists():
            return adapter

        parent = folder.parent
        if str(parent) == '/':
            return

        return cls.find_adapter(parent)


    @classmethod
    def from_folder(cls,
            folder: PathLike, adapter: PathLike = None, meta: UserMetaData = None,
            serial: str = None,
        ) -> "TemplatedDataSet":
        """Constructs a templated dataset from a folder.

        Args:
            folder: path to the folder where the data files are stored.
            adapter: the adapter file to create group structure from.
            meta: user metadata to use in time localization when loading data. If not
                provided, then the datasets will *not* be loaded.
        """
        if (    serial is not None and
                meta is not None and
                meta.dserial != serial and
                meta.datafile.serial != serial
            ):
            return

        folder = Path(folder).expanduser()
        if adapter is None and meta is not None:
            adapter = meta.ADAPTER_FILEPATH

        if adapter is None:
            adapter = cls.find_adapter(folder)
            if adapter is None:
                raise ValueError(
                    f"Unable to create a templated data set; no adapter file found in {folder}."
                )

        result = cls(adapter)
        for group in result.groups:
            try:
                result.create_group(group)
            except:
                log.error(f"While creating dynamic properties for group {group}.", exc_info=True)
                raise

        cls.TIME_SERIES = list(result.times.values())

        if meta is not None:
            try:
                result.load_group_data(folder, meta=meta)
            except:
                log.error(
                    f"While loading group data from {folder} for {meta.user}/{meta.dserial}.",
                    exc_info=True,
                )

        return result


    @classmethod
    def from_file(cls, file: PathLike, serial: str = None) -> "TemplatedDataSet":
        """Extracts a data set for a single user from a single file.
        """
        target = Path(file).expanduser()
        return cls.from_folder(target.parent, serial=serial)


    def mask(self,
            regions: List[Tuple[float, float]] = None,
            lbuffer: float = 5.0,
            rbuffer: float = 5.0,
        ):
        """Mask the outputs of this data source so that only those samples
        within the `regions` are streamed/returned.

        Args:
            regions: list of `(start, stop)` times that define the boundaries
                to *keep* in the data source.
        """
        self.mask_regions = regions
        self.mask_lbuffer = lbuffer
        self.mask_rbuffer = rbuffer
        self.mask_exclude = False

        self._reslice()


    def resample(self, **kwargs):
        """Set resampling options for each of the sensors.

        Args:
            kwargs: key-value pairs specifying the `{group}_{prop}` to set
                resampling options for. Set to `None` to cancel the resampling
                options. Values should be :class:`ResamplingOptions` instances.
        """
        self.resampling.update(kwargs)
        self._reslice()
