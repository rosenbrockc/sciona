"""Standalone templated dataset module for the principal evaluator.

Provides adapter.yml-driven multi-group dataset loading.
"""

from .core import (
    TemplatedDataSet,
    DataFileName,
    ZeroSourceFilesError,
    run_vector_property,
    merge_templated_group,
    time_merge_dataframes,
    get_source_hash,
    get_prop_name,
    get_datafile_exts,
    datafile_from_spec,
    get_templated_subfolders,
    make_uday_label,
)
from .factories import (
    create_templated_meta_class,
    create_templated_dataset_collection,
)
from .io import read as read_adapter

__all__ = [
    "TemplatedDataSet",
    "DataFileName",
    "ZeroSourceFilesError",
    "run_vector_property",
    "merge_templated_group",
    "time_merge_dataframes",
    "get_source_hash",
    "get_prop_name",
    "get_datafile_exts",
    "datafile_from_spec",
    "get_templated_subfolders",
    "make_uday_label",
    "create_templated_meta_class",
    "create_templated_dataset_collection",
    "read_adapter",
]
