"""Factory functions for dynamically creating templated metadata and collection classes.

Extracted from ``ageom.datasets.templated``, retaining only the generic
adapter-driven dataset machinery.
"""

from pathlib import Path
from os import PathLike, path
from typing import List, Union, Mapping, Callable
from dataclasses import make_dataclass, field, fields
import re
import logging
import numpy as np
import pandas as pd

from ageom.datasets.parser.base import (
    DataSetBase,
    UserMetaData,
    DataSetCollection,
    FolderFilterOptions,
    UserDay,
)
from ageom.datasets.parser.utils import import_fqn, execute_any_transform

from .core import (
    DataFileName,
    TemplatedDataSet,
    datafile_from_spec,
    get_datafile_exts,
    get_templated_subfolders,
    make_uday_label,
    _get_datafile_any,
    ZeroSourceFilesError,
    METACLASSES,
    COLLECTION_CLASSES,
    PER_PARSER_CACHE,
    PARSER_CACHE,
    DEFAULT_UDAY_LABEL,
    msg,
)
from .io import read

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stub file source (replaces create_templated_file_source)
# ---------------------------------------------------------------------------

class _StubFileSource:
    """Satisfies ``UserMetaData``'s ``SOURCE`` slot without any remote
    file discovery capabilities.
    """
    @classmethod
    def matcher(cls):
        return None

    @classmethod
    def parse(cls, filepath):
        return None


# ---------------------------------------------------------------------------
# create_templated_meta_class
# ---------------------------------------------------------------------------

def create_templated_meta_class(
        template_path: PathLike, template: dict = None, varset: dict = None,
    ) -> UserMetaData:
    """Creates a subclass of :class:`UserMetaData` that includes the
    customizations specified in the template ``adapter.yml`` file.

    Args:
        template_path: path to the specification for grabbing the user metadata from either
            files or the naming of folders and files.
        template: if the template has already been loaded from file, then prevent
            reloading by passing it here.
        varset: if any template values should have variable values substituted, specify
            the key-value mappings here.
    """
    template_file = Path(template_path).expanduser()
    if template_file in METACLASSES:
        return METACLASSES[template_file]

    if template is None:
        template = read(
            template_file.parent, path.splitext(template_file.name)[0], varset=varset,
        )

    tname, mspec = template["name"], template["meta"]
    classname = f"{tname}UserMetaData"

    def __init__(self,
        filepath: PathLike, folder: PathLike = None, datafile: DataFileName = None,
        cache: PathLike = None, metadata: dict = None, metaspec: dict = None,
        connections: list = None, extras: List[str] = None,
    ):
        """Constructs a templated version of the :class:`UserMetaData` that was generated
        dynamically. See the super class for argument documentation.

        Args:
            metaspec: reference to the `meta` section of the dataset template.
        """
        UserMetaData.__init__(self,
            filepath, folder, datafile, cache, metadata, connections=connections,
        )
        self.metaspec = metaspec

        self.device_cls = None
        if "device" in metaspec:
            dfields = [(s["name"], field(default=None)) for s in metaspec["device"]]
            fieldoc = '\n'.join([
                f"{s['name']}: {s['description']}" for s in metaspec["device"]
            ])
            namespace = {}

            if any("transform" in s for s in metaspec["device"]):
                def __post_init__(self):
                    for s in metaspec["device"]:
                        if "transform" in s:
                            value = execute_any_transform(s, getattr(self, s["name"]))
                            setattr(self, s["name"], value)

                namespace["__post_init__"] = __post_init__

            self.device_cls = make_dataclass(
                f"{tname.title()}DeviceMetaData", dfields,
                namespace=namespace,
            )
            self.device_cls.__doc__ = """Auto-generated device metadata class for template {tname}.

            Attributes:
                {fieldoc}
            """.format(tname=tname, fieldoc=fieldoc)


    @classmethod
    def from_group_data(cls,
            filepath: PathLike, data: dict, folder: PathLike = None
        ) -> UserMetaData:
        """Constructs a new instance of the {classname} metadata using a single
        row of metadata from a group file.
        """.format(classname=classname)
        metaspec = cls.TEMPLATE["meta"]
        datafile = datafile_from_spec(metaspec, data)
        if folder is None:
            folder = path.dirname(filepath)

        # No data connections in the standalone module.
        connections = []

        return cls(
            filepath, folder, datafile, metadata=data, metaspec=metaspec,
            connections=connections,
        )


    @classmethod
    def from_file(cls, filepath: PathLike, folder: PathLike = None) -> Union[UserMetaData, List[UserMetaData]]:
        """Constructs a new instance of the {classname} metadata using a metadata
        file located in a single folder of trial/user day data.
        """.format(classname=classname)
        filepath = Path(filepath).expanduser()
        if folder is not None:
            folder = Path(folder).expanduser()
        metaspec = cls.TEMPLATE["meta"]

        reader, kwargs = metaspec["reader"]["fqn"], metaspec["reader"].get("kwargs", {})
        _, caller = import_fqn(reader)
        try:
            metadata = caller(str(filepath), **kwargs)
        except:
            log.error(f"While parsing metadata file at {filepath}; critical error.", exc_info=True)
            msg.err(f"While parsing metadata file at {filepath}; critical error.")
            raise

        if "transform" in metaspec["reader"]:
            metadata = execute_any_transform(
                metaspec["reader"], metadata, metaspec,
            )

        if metaspec.get("per", True):
            return cls.from_group_data(filepath, metadata, folder)
        else:
            dirattr = metaspec.get("folder")
            if folder is None:
                folder = cls.ADAPTER_FILEPATH.parent
            result = []

            for ridx in range(len(metadata)):
                row = metadata.iloc[ridx].to_dict()
                _folder = folder
                if dirattr is not None:
                    filename = _get_datafile_any(metaspec, row, "folder")
                    _folder = folder.joinpath(filename)

                result.append(cls.from_group_data(filepath, row, _folder))

            return result


    def device(self):
        """Gets metadata specific to the device being used.
        """
        if self.device_cls is not None:
            fnames = [f.name for f in fields(self.device_cls)]
            relevant = {k: v for k, v in self.metadata if k in fnames}
            return self.device_cls(**relevant)


    def user_id(self) -> str:
        """Gets the user id from the metadata file contents, which may be
        different from :prop:`user`, which comes from the file name convention.
        """
        if "user" in self.metaspec:
            return self.metadata.get(self.metaspec["user"])


    def is_valid(self) -> bool:
        """Check if the current data file passes our validity criteria.
        """
        if "validator" in self.metaspec:
            return execute_any_transform(self.metaspec["validator"], self.data, self.metadata)


    namespace = {
        "__init__": __init__,
        "device": property(device),
        "user_id": property(user_id),
        "is_valid": property(is_valid),
        "ADAPTER_FILEPATH": template_file,
        "TEMPLATE": template,
        "META_EXT": path.splitext(mspec["source"])[1],
        "DATA_EXT": get_datafile_exts(template, first=False),
        "SOURCE": _StubFileSource,
        "DATASET": type(f"{tname}TemplatedDataSet", (TemplatedDataSet,), {}),
        "from_group_data": from_group_data,
        "from_file": from_file,
    }

    if "other" in mspec:
        for s in mspec["other"]:
            def propfun(self, s=s):
                f"""{s['description']}"""
                value = self.metadata[s["name"]]
                if "transform" in s:
                    return execute_any_transform(
                        s["transform"], value, self.metadata
                    )
                return value

            namespace[s["name"]] = property(propfun)


    cls = type(
        classname, (UserMetaData,), namespace,
    )
    cls.__doc__ = f"""Auto-generated template metadata from template {tname}.
    """
    METACLASSES[template_file] = cls

    return METACLASSES[template_file]


# ---------------------------------------------------------------------------
# create_templated_dataset_collection
# ---------------------------------------------------------------------------

def create_templated_dataset_collection(
        template_path: PathLike, template: dict = None, varset: dict = None,
    ) -> DataSetCollection:
    """Creates a subclass of :class:`DataSetCollection` that includes the
    customizations specified in the template ``adapter.yml`` file.

    Args:
        template_path: path to the specification for grabbing the user metadata from either
            files or the naming of folders and files.
        template: if the template has already been loaded from file, then prevent
            reloading by passing it here.
        varset: if any template values should have variable values substituted, specify
            the key-value mappings here.
    """
    template_file = Path(template_path).expanduser()
    if template_file in COLLECTION_CLASSES:
        return COLLECTION_CLASSES[template_file]

    if template is None:
        template = read(
            template_file.parent, path.splitext(template_file.name)[0], varset=varset,
        )

    tname, mspec = template["name"], template["meta"]
    metacls = create_templated_meta_class(template_path, template)
    classname = f"{tname}DataSetCollection"


    def parser(folder: PathLike, regex: re.Pattern, data_ext: str,
                cls: DataSetCollection, recursive: bool = False) -> List[UserMetaData]:
        """Custom folder parser for templated dataset collections.
        """
        folder = Path(folder).expanduser()
        if mspec.get("per", True):
            if folder in PER_PARSER_CACHE:
                return PER_PARSER_CACHE[folder]
        else:
            filepath = folder.joinpath(mspec["source"])
            if filepath in PARSER_CACHE:
                return PARSER_CACHE[filepath]

        if mspec.get("per", True):
            targets = get_templated_subfolders(folder, regex, recursive)
            files: List[UserMetaData] = []
            for target in targets:
                files.append(metacls.from_file(target.joinpath(mspec["source"]), target))
        else:
            filepath = folder.joinpath(mspec["source"])
            files: List[UserMetaData] = metacls.from_file(filepath, folder)

        result = {}
        for f in files:
            df_user = str(f.datafile.user)
            if df_user not in result:
                result[df_user] = []
            result[df_user].append(f)

        if mspec.get("per", True):
            PER_PARSER_CACHE[folder] = result
        else:
            PARSER_CACHE[filepath] = result

        return result


    @classmethod
    def from_folder(cls,
            folder: PathLike = None, options: FolderFilterOptions = None, label: str = None
        ) -> "DataSetCollection":
        """Get a user dataset for multiple dataset files from a folder.

        Args:
            options: filtering options for when a folder contains multiple
                files across multiple devices and data types for a single user.
                If not specified, then this will default to the first user, and
                the user's first device.
            label: a user-specified label to represent the data in this user day. For
                human-consumption/UX only.
        """
        return DataSetCollection.from_folder(
            cls.META_CLASS.ADAPTER_FILEPATH.parent, subcls=cls, options=options, label=label,
        )


    def autoload(self, uday: UserDay = None, folder: PathLike = None):
        """Merges multiple data sources ensuring time is sorted.

        Args:
            uday: optional user day to select metadata for merge ordering.
            folder: if you want to autoload from a non-default folder for the
                template file, specify that here.
        """
        if len(self.ordered) == 0:
            raise ZeroSourceFilesError(folder)

        sources = [m.data for m in self.ordered]
        if len(sources) == 1:
            self.data = sources[0]
        else:
            meta = self.meta if uday is None else self.get_meta(uday)
            first: TemplatedDataSet = sources[0]
            self.data = first.merge(*sources,
                meta=meta, srcmeta=self.ordered, folder=folder,
            )

        self.data.slice(self._start, self._stop)
        self.data.mask(self.mask_regions, self.mask_lbuffer, self.mask_rbuffer)
        self.data._reslice()
        return self.data


    def to_pandas(self,
            loader: Callable[[UserMetaData], DataSetBase] = None,
            merger: Callable[[List[DataSetBase]], DataSetBase] = None,
            uday: UserDay = None,
        ) -> Mapping[str, pd.DataFrame]:
        """Converts this dataset collection into a set of dataframes grouped
        by the sensor groups in the templated dataset.
        """
        self.autoload(uday)
        return DataSetCollection.to_pandas(self, exclude=("*_start", "*_stop"))


    namespace = {
        "DATA_CLASS": TemplatedDataSet,
        "META_CLASS": metacls,
        "CUSTOM_PARSER": parser,
        "VENDORS": [],
        "HAPPY": None,
        "from_folder": from_folder,
        "autoload": autoload,
        "to_pandas": to_pandas,
    }

    cls = type(
        classname, (DataSetCollection,), namespace,
    )
    cls.__doc__ = f"""Auto-generated dataset collection from template {tname}.
    """
    COLLECTION_CLASSES[template_file] = cls

    return COLLECTION_CLASSES[template_file]
