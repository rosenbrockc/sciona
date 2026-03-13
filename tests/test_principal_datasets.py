"""Tests for ageom.principal.datasets — the standalone templated dataset module."""

from __future__ import annotations

import json
import re
import textwrap
from datetime import datetime, date
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# io.py
# ---------------------------------------------------------------------------
from ageom.principal.datasets.io import (
    is_link,
    substitute_varset,
    read,
    LINK_CHARACTER,
    REF_CHARACTER,
)


class TestIsLink:
    def test_local_link(self):
        assert is_link(":some/path") is True

    def test_ref_link(self):
        assert is_link("^studio/resource") is True

    def test_plain_string(self):
        assert is_link("plain") is False

    def test_empty_string(self):
        assert is_link("") is False

    def test_non_string(self):
        assert is_link(42) is False
        assert is_link(None) is False
        assert is_link(["a"]) is False


class TestSubstituteVarset:
    def test_single_substitution(self):
        assert substitute_varset("file_$(name).csv", {"name": "test"}) == "file_test.csv"

    def test_multiple_substitutions(self):
        result = substitute_varset("$(a)_$(b)", {"a": "x", "b": "y"})
        assert result == "x_y"

    def test_no_varset(self):
        assert substitute_varset("$(name)", None) == "$(name)"

    def test_no_dollar_sign(self):
        assert substitute_varset("nothing_here", {"name": "val"}) == "nothing_here"

    def test_unmatched_var_left_alone(self):
        assert substitute_varset("$(missing)", {"other": "val"}) == "$(missing)"


class TestRead:
    def test_reads_simple_yaml(self, tmp_path):
        adapter = tmp_path / "adapter.yml"
        adapter.write_text("name: test\nvalue: 42\n")
        result = read(str(tmp_path), "adapter")
        assert result == {"name": "test", "value": 42}

    def test_varset_substitution(self, tmp_path):
        adapter = tmp_path / "data.yml"
        adapter.write_text("source: tracker_$(tracker).csv\n")
        result = read(str(tmp_path), "data", varset={"tracker": "full"})
        assert result == {"source": "tracker_full.csv"}

    def test_local_link_resolution(self, tmp_path):
        child = tmp_path / "child.yml"
        child.write_text("key: linked_value\n")
        parent = tmp_path / "parent.yml"
        parent.write_text("nested: ':child'\n")
        result = read(str(tmp_path), "parent")
        assert result["nested"] == {"key": "linked_value"}

    def test_ref_character_raises(self, tmp_path):
        adapter = tmp_path / "adapter.yml"
        adapter.write_text("ref: '^studio/resource'\n")
        with pytest.raises(NotImplementedError, match="Remote studio references"):
            read(str(tmp_path), "adapter")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="was not found"):
            read(str(tmp_path), "nonexistent")

    def test_non_recursive_link_returns_empty(self, tmp_path):
        adapter = tmp_path / "adapter.yml"
        adapter.write_text("nested: ':child'\n")
        result = read(str(tmp_path), "adapter", recursive=False)
        # Non-recursive: links are not followed, so the link stays as-is
        # Actually, the link character triggers read() internally which returns {}
        assert result["nested"] == {}


# ---------------------------------------------------------------------------
# core.py — DataFileName
# ---------------------------------------------------------------------------
from ageom.principal.datasets.core import DataFileName


class TestDataFileName:
    def test_basic_creation(self):
        df = DataFileName("user1", "serial1", datetime(2024, 3, 15, 10, 30))
        assert df.user == "user1"
        assert df.serial == "serial1"
        assert df.start == datetime(2024, 3, 15, 10, 30)
        assert df.stop is None

    def test_day_property_from_datetime(self):
        df = DataFileName("u", "s", datetime(2024, 6, 20, 14, 0))
        assert df.day == date(2024, 6, 20)

    def test_day_property_from_date(self):
        d = date(2024, 6, 20)
        df = DataFileName("u", "s", d)
        assert df.day == d

    def test_with_stop(self):
        start = datetime(2024, 1, 1)
        stop = datetime(2024, 1, 2)
        df = DataFileName("u", "s", start, stop)
        assert df.stop == stop


# ---------------------------------------------------------------------------
# core.py — helper functions
# ---------------------------------------------------------------------------
from ageom.principal.datasets.core import (
    get_prop_name,
    get_source_hash,
    get_datafile_exts,
    time_merge_dataframes,
    run_vector_property,
    datafile_from_spec,
    _get_datafile_day,
    _get_datafile_any,
    make_uday_label,
    get_templated_subfolders,
    ZeroSourceFilesError,
)


class TestGetPropName:
    def test_regular_property(self):
        assert get_prop_name("accel", "x") == "accel_x"

    def test_bracket_property(self):
        assert get_prop_name("accel", "[]") == "accel"

    def test_private(self):
        assert get_prop_name("accel", "x", private=True) == "_accel_x"

    def test_private_bracket(self):
        assert get_prop_name("accel", "[]", private=True) == "_accel"


class TestGetSourceHash:
    def test_same_input_same_hash(self):
        g = {"source": "*.csv", "reader": {"fqn": "pandas.read_csv"}, "extra": "ignored"}
        assert get_source_hash(g) == get_source_hash(g)

    def test_different_source_different_hash(self):
        g1 = {"source": "a.csv", "reader": {"fqn": "pandas.read_csv"}}
        g2 = {"source": "b.csv", "reader": {"fqn": "pandas.read_csv"}}
        assert get_source_hash(g1) != get_source_hash(g2)

    def test_ignores_non_source_reader_keys(self):
        g1 = {"source": "a.csv", "reader": {"fqn": "r"}, "time": "t"}
        g2 = {"source": "a.csv", "reader": {"fqn": "r"}, "time": "x"}
        assert get_source_hash(g1) == get_source_hash(g2)


class TestGetDatafileExts:
    def test_first_extension(self):
        t = {"groups": {"ppg": {"source": "ring/*.HPY2"}}}
        assert get_datafile_exts(t, first=True) == "HPY2"

    def test_all_extensions(self):
        t = {"groups": {
            "ppg": {"source": "*.csv"},
            "accel": {"source": "*.parquet"},
        }}
        assert get_datafile_exts(t, first=False) == ["csv", "parquet"]

    def test_no_groups_raises(self):
        with pytest.raises(KeyError, match="does not contain any groups"):
            get_datafile_exts({})

    def test_no_source_raises(self):
        with pytest.raises(KeyError, match="does not have a source"):
            get_datafile_exts({"groups": {"g": {}}}, first=True)


class TestTimeMergeDataframes:
    def test_merges_in_time_order(self):
        df1 = pd.DataFrame({"t": [10.0, 11.0], "v": [1, 2]})
        df2 = pd.DataFrame({"t": [5.0, 6.0], "v": [3, 4]})
        result = time_merge_dataframes([df1, df2], "t")
        assert list(result["t"]) == [5.0, 6.0, 10.0, 11.0]
        assert list(result.index) == [0, 1, 2, 3]

    def test_skips_empty_dataframes(self):
        df1 = pd.DataFrame({"t": [1.0], "v": [10]})
        df2 = pd.DataFrame({"t": [], "v": []})
        result = time_merge_dataframes([df1, df2], "t")
        assert len(result) == 1

    def test_single_dataframe(self):
        df = pd.DataFrame({"t": [3.0, 4.0], "v": [5, 6]})
        result = time_merge_dataframes([df], "t")
        assert list(result["v"]) == [5, 6]


class TestRunVectorProperty:
    def test_basic_slicing(self):
        source = np.array([10, 20, 30, 40, 50])
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cast = {}
        result = run_vector_property("val", "t", cast, source, time, 1, 4, {})
        np.testing.assert_array_equal(result, [20, 30, 40])
        assert "val" in cast

    def test_caches_result(self):
        cast = {"val": np.array([99])}
        result = run_vector_property("val", "t", cast, np.array([1]), np.array([0]), 0, 1, {})
        np.testing.assert_array_equal(result, [99])

    def test_none_source_returns_empty(self):
        cast = {}
        result = run_vector_property("val", "t", cast, None, np.array([]), 0, 0, {})
        assert len(result) == 0

    def test_mask_applied(self):
        source = np.array([10, 20, 30, 40])
        time = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([True, False, True, False])
        cast = {"t_mask": mask}
        result = run_vector_property("val", "t", cast, source, time, 0, 4, {})
        np.testing.assert_array_equal(result, [10, 30])


class TestDatafileFromSpec:
    def test_basic_extraction(self):
        spec = {
            "day": {"source": "date_col", "ftime": "%Y-%m-%d"},
            "serial": {"source": "serial_col"},
            "user": {"source": "user_col"},
        }
        data = {"date_col": "2024-03-15", "serial_col": "SN001", "user_col": "alice"}
        result = datafile_from_spec(spec, data)
        assert isinstance(result, DataFileName)
        assert result.user == "alice"
        assert result.serial == "SN001"
        assert result.day == date(2024, 3, 15)

    def test_missing_day_source_raises(self):
        spec = {"day": {"source": "missing"}, "serial": {"source": "s"}, "user": {"source": "u"}}
        with pytest.raises(ValueError, match="missing"):
            datafile_from_spec(spec, {"s": "x", "u": "y"})


class TestGetDatafileDay:
    def test_default_format(self):
        spec = {"day": {"source": "d"}}
        result = _get_datafile_day(spec, {"d": "03/15/24"})
        assert result == date(2024, 3, 15)

    def test_custom_format(self):
        spec = {"day": {"source": "d", "ftime": "%Y-%m-%d"}}
        result = _get_datafile_day(spec, {"d": "2024-03-15"})
        assert result == date(2024, 3, 15)

    def test_no_source_no_transform_raises(self):
        spec = {"day": {}}
        with pytest.raises(ValueError, match="source.*transform"):
            _get_datafile_day(spec, {})


class TestGetDatafileAny:
    def test_extracts_value(self):
        spec = {"serial": {"source": "sn"}}
        assert _get_datafile_any(spec, {"sn": "ABC"}, "serial") == "ABC"

    def test_missing_source_raises(self):
        spec = {"serial": {"source": "missing"}}
        with pytest.raises(ValueError, match="missing"):
            _get_datafile_any(spec, {}, "serial")


class TestMakeUdayLabel:
    def test_default_format(self):
        df = DataFileName("alice", "SN01", datetime(2024, 1, 15))
        result = make_uday_label("{user}/{serial}/{day}", df, {})
        assert result == "alice/SN01/2024-01-15 00:00:00"

    def test_extra_metadata_fields(self):
        df = DataFileName("bob", "SN02", datetime(2024, 6, 1))
        result = make_uday_label("{user}-{trial}", df, {"trial": "T1", "user": "ignored"})
        assert result == "bob-T1"


class TestGetTemplatedSubfolders:
    def test_no_regex_returns_all(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "file.txt").touch()
        result = get_templated_subfolders(tmp_path, None)
        names = {p.name for p in result}
        assert names == {"a", "b"}

    def test_regex_filters(self, tmp_path):
        (tmp_path / "trial_01").mkdir()
        (tmp_path / "trial_02").mkdir()
        (tmp_path / "other").mkdir()
        rx = re.compile(r"trial_\d+")
        result = get_templated_subfolders(tmp_path, rx)
        names = {p.name for p in result}
        assert names == {"trial_01", "trial_02"}

    def test_recursive(self, tmp_path):
        child = tmp_path / "a"
        child.mkdir()
        grandchild = child / "b"
        grandchild.mkdir()
        result = get_templated_subfolders(tmp_path, None, recursive=True)
        names = {p.name for p in result}
        assert names == {"a", "b"}


class TestZeroSourceFilesError:
    def test_message_and_folder(self):
        err = ZeroSourceFilesError(Path("/data/test"))
        assert "/data/test" in str(err)
        assert err.folder == Path("/data/test")

    def test_is_value_error(self):
        assert issubclass(ZeroSourceFilesError, ValueError)


# ---------------------------------------------------------------------------
# core.py — TemplatedDataSet with a real adapter on disk
# ---------------------------------------------------------------------------
from ageom.principal.datasets.core import TemplatedDataSet


def _write_csv_adapter(tmp_path: Path) -> Path:
    """Create a minimal adapter.yml + CSV data for testing TemplatedDataSet."""
    adapter = tmp_path / "adapter.yml"
    adapter.write_text(textwrap.dedent("""\
        name: TestSet
        groups:
          sensor:
            reader:
              fqn: pandas.read_csv
            source: "data.csv"
            time: t
            properties:
              t:
                source: timestamp
                description: Epoch seconds.
              value:
                source: reading
                description: Sensor reading.
        meta:
          source: meta.json
          per: True
          reader:
            fqn: json.load
          day:
            source: date
            ftime: "%Y-%m-%d"
          serial:
            source: serial
          user:
            source: user
    """))
    data = tmp_path / "data.csv"
    data.write_text("timestamp,reading\n1.0,10\n2.0,20\n3.0,30\n")
    return adapter


class TestTemplatedDataSet:
    def test_init_parses_groups(self, tmp_path):
        adapter = _write_csv_adapter(tmp_path)
        ds = TemplatedDataSet(adapter)
        assert "sensor" in ds.groups
        assert ds.template["name"] == "TestSet"

    def test_init_missing_groups_raises(self, tmp_path):
        bad = tmp_path / "bad.yml"
        bad.write_text("name: NoGroups\nmeta: {}\n")
        with pytest.raises(TypeError, match="does not define any groups"):
            TemplatedDataSet(bad)

    def test_create_group(self, tmp_path):
        adapter = _write_csv_adapter(tmp_path)
        ds = TemplatedDataSet(adapter)
        ds.create_group("sensor")
        assert ds.times["sensor"] == "sensor_t"
        assert ds.rtimes["sensor"] == "_sensor_t"
        assert ds.starts["sensor"] == "sensor_start"
        assert ds.stops["sensor"] == "sensor_stop"

    def test_load_group_data(self, tmp_path):
        adapter = _write_csv_adapter(tmp_path)
        ds = TemplatedDataSet(adapter)
        ds.create_group("sensor")
        ds.load_group_data(folder=tmp_path)
        np.testing.assert_array_equal(ds._sensor_t, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(ds._sensor_value, [10, 20, 30])

    def test_load_missing_source_returns_empty_df(self, tmp_path):
        adapter = _write_csv_adapter(tmp_path)
        # Delete the data file so source pattern won't match
        (tmp_path / "data.csv").unlink()
        ds = TemplatedDataSet(adapter)
        ds.create_group("sensor")
        ds.load_group_data(folder=tmp_path)
        assert len(ds._sensor_t) == 0

    def test_load_group_data_skips_failed_group(self, tmp_path, monkeypatch):
        adapter = tmp_path / "adapter.yml"
        adapter.write_text(textwrap.dedent("""\
            name: FaultTolerant
            groups:
              sensor:
                reader:
                  fqn: pandas.read_csv
                source: data.csv
                time: t
                properties:
                  t: {source: t}
                  value: {source: value}
              broken:
                reader:
                  fqn: pandas.read_csv
                source: missing.csv
                time: t
                properties:
                  t: {source: t}
                  value: {source: value}
        """))
        (tmp_path / "data.csv").write_text("t,value\n1,10\n2,20\n")

        ds = TemplatedDataSet(adapter)
        ds.create_group("sensor")
        ds.create_group("broken")

        real_load = ds.load

        def _fake_load(group, folder, meta=None):
            if group == "broken":
                raise ValueError("boom")
            return real_load(group, folder, meta=meta)

        monkeypatch.setattr(ds, "load", _fake_load)
        ds.load_group_data(folder=tmp_path)

        np.testing.assert_array_equal(ds._sensor_t, [1.0, 2.0])
        np.testing.assert_array_equal(ds._sensor_value, [10, 20])
        assert len(ds._broken_t) == 0
        assert len(ds._broken_value) == 0

    def test_from_folder(self, tmp_path):
        adapter = _write_csv_adapter(tmp_path)
        ds = TemplatedDataSet.from_folder(tmp_path, adapter=adapter)
        assert "sensor" in ds.groups
        assert ds.times["sensor"] == "sensor_t"

    def test_find_adapter(self, tmp_path):
        adapter = _write_csv_adapter(tmp_path)
        child = tmp_path / "subdir"
        child.mkdir()
        found = TemplatedDataSet.find_adapter(child)
        assert found == adapter

    def test_find_adapter_not_found(self, tmp_path):
        result = TemplatedDataSet.find_adapter(tmp_path)
        assert result is None

    def test_dynamic_property_access(self, tmp_path):
        """After load, group properties should be accessible and sliceable."""
        adapter = _write_csv_adapter(tmp_path)
        ds = TemplatedDataSet.from_folder(tmp_path, adapter=adapter)
        ds.load_group_data(folder=tmp_path)
        ds._reslice()
        t = ds.sensor_t
        np.testing.assert_array_equal(t, [1.0, 2.0, 3.0])
        v = ds.sensor_value
        np.testing.assert_array_equal(v, [10, 20, 30])

    def test_duration(self, tmp_path):
        adapter = _write_csv_adapter(tmp_path)
        ds = TemplatedDataSet.from_folder(tmp_path, adapter=adapter)
        ds.load_group_data(folder=tmp_path)
        ds._reslice()
        assert ds.duration == pytest.approx(2.0)

    def test_min_max(self, tmp_path):
        adapter = _write_csv_adapter(tmp_path)
        ds = TemplatedDataSet.from_folder(tmp_path, adapter=adapter)
        ds.load_group_data(folder=tmp_path)
        ds._reslice()
        assert ds.min == pytest.approx(1.0)
        assert ds.max == pytest.approx(3.0)

    def test_varset_substitution(self, tmp_path):
        adapter = tmp_path / "adapter.yml"
        adapter.write_text(textwrap.dedent("""\
            name: VarTest
            groups:
              sensor:
                reader:
                  fqn: pandas.read_csv
                source: "$(prefix)_data.csv"
                time: t
                properties:
                  t:
                    source: timestamp
                    description: time
            meta:
              source: meta.json
              per: True
              reader:
                fqn: json.load
              day:
                source: date
                ftime: "%Y-%m-%d"
              serial:
                source: serial
              user:
                source: user
        """))
        ds = TemplatedDataSet(adapter, varset={"prefix": "sensor"})
        assert ds.groups["sensor"]["source"] == "sensor_data.csv"


# ---------------------------------------------------------------------------
# factories.py
# ---------------------------------------------------------------------------
from ageom.principal.datasets.factories import (
    create_templated_meta_class,
    create_templated_dataset_collection,
    _StubFileSource,
)
from ageom.principal.datasets.core import METACLASSES, COLLECTION_CLASSES


class TestStubFileSource:
    def test_matcher_returns_none(self):
        assert _StubFileSource.matcher() is None

    def test_parse_returns_none(self):
        assert _StubFileSource.parse("/some/path") is None


class TestCreateTemplatedMetaClass:
    def test_creates_named_class(self, tmp_path):
        adapter = _write_csv_adapter(tmp_path)
        # Clear cache to avoid cross-test contamination
        METACLASSES.pop(adapter, None)
        cls = create_templated_meta_class(str(adapter))
        assert cls.__name__ == "TestSetUserMetaData"

    def test_caches_class(self, tmp_path):
        adapter = _write_csv_adapter(tmp_path)
        METACLASSES.pop(adapter, None)
        cls1 = create_templated_meta_class(str(adapter))
        cls2 = create_templated_meta_class(str(adapter))
        assert cls1 is cls2

    def test_source_is_stub(self, tmp_path):
        adapter = _write_csv_adapter(tmp_path)
        METACLASSES.pop(adapter, None)
        cls = create_templated_meta_class(str(adapter))
        assert cls.SOURCE is _StubFileSource

    def test_template_and_adapter_path_stored(self, tmp_path):
        adapter = _write_csv_adapter(tmp_path)
        METACLASSES.pop(adapter, None)
        cls = create_templated_meta_class(str(adapter))
        assert cls.ADAPTER_FILEPATH == adapter
        assert cls.TEMPLATE["name"] == "TestSet"


class TestCreateTemplatedDatasetCollection:
    def test_creates_named_class(self, tmp_path):
        adapter = _write_csv_adapter(tmp_path)
        METACLASSES.pop(adapter, None)
        COLLECTION_CLASSES.pop(adapter, None)
        cls = create_templated_dataset_collection(str(adapter))
        assert cls.__name__ == "TestSetDataSetCollection"

    def test_caches_class(self, tmp_path):
        adapter = _write_csv_adapter(tmp_path)
        METACLASSES.pop(adapter, None)
        COLLECTION_CLASSES.pop(adapter, None)
        cls1 = create_templated_dataset_collection(str(adapter))
        cls2 = create_templated_dataset_collection(str(adapter))
        assert cls1 is cls2

    def test_happy_is_none(self, tmp_path):
        adapter = _write_csv_adapter(tmp_path)
        METACLASSES.pop(adapter, None)
        COLLECTION_CLASSES.pop(adapter, None)
        cls = create_templated_dataset_collection(str(adapter))
        assert cls.HAPPY is None

    def test_vendors_is_empty(self, tmp_path):
        adapter = _write_csv_adapter(tmp_path)
        METACLASSES.pop(adapter, None)
        COLLECTION_CLASSES.pop(adapter, None)
        cls = create_templated_dataset_collection(str(adapter))
        assert cls.VENDORS == []


# ---------------------------------------------------------------------------
# __init__.py — public API surface
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_all_symbols_importable(self):
        from ageom.principal.datasets import (
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
            create_templated_meta_class,
            create_templated_dataset_collection,
            read_adapter,
        )

    def test_read_adapter_alias(self):
        from ageom.principal.datasets import read_adapter
        assert read_adapter is read


# ---------------------------------------------------------------------------
# No HPY references
# ---------------------------------------------------------------------------

class TestNoHPYReferences:
    """Ensure the module source files contain zero HPY references in code."""

    @pytest.fixture
    def module_sources(self):
        base = Path(__file__).resolve().parent.parent / "ageom" / "principal" / "datasets"
        return [
            base / "__init__.py",
            base / "io.py",
            base / "core.py",
            base / "factories.py",
        ]

    def test_no_hpy_in_code_lines(self, module_sources):
        """Every line that isn't an 'import ... as' alias should be HPY-free."""
        for src in module_sources:
            for lineno, line in enumerate(src.read_text().splitlines(), 1):
                stripped = line.strip()
                # Skip import alias lines (the only acceptable reference)
                if "as DataFileName" in stripped:
                    continue
                assert "HPY" not in stripped and "hpy" not in stripped.lower() or "hpy" in "unhappy", (
                    f"{src.name}:{lineno} contains HPY reference: {stripped}"
                )


# ---------------------------------------------------------------------------
# evaluator.py — evaluate_adapter (lazy import)
# ---------------------------------------------------------------------------

class TestEvaluateAdapterLazyImport:
    """The datasets import must not happen at evaluator module load time."""

    def test_evaluator_importable_without_datasets(self):
        """Importing ExecutionSandbox should not fail even if datasets
        internals have issues — the import is lazy."""
        from ageom.principal.evaluator import ExecutionSandbox
        sandbox = ExecutionSandbox(timeout_s=5.0)
        assert sandbox._timeout_s == 5.0


class TestEvaluateAdapterMethod:
    @pytest.fixture
    def sandbox(self):
        from ageom.principal.evaluator import ExecutionSandbox
        return ExecutionSandbox(timeout_s=10.0)

    def test_missing_adapter_returns_penalty(self, sandbox, tmp_path):
        from ageom.principal.models import OptimizationMetric
        from ageom.synthesizer.models import ExportBundle

        bundle = MagicMock(spec=ExportBundle)
        bundle.output_dir = tmp_path

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            sandbox.evaluate_adapter(
                bundle,
                str(tmp_path / "nonexistent_adapter.yml"),
                OptimizationMetric.LATENCY,
            )
        )
        assert result.global_loss == 1e12
