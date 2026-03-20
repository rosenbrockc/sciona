from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sciona.principal.adapters.parsers import (
    CAPNO_COLS,
    parse_capnostream_file,
    polar_h10_to_pandas,
)
from sciona.principal.adapters.transforms import shift_time_meta_attr


def _capno_row(date: str, time: str, rr: str) -> str:
    values = {name: "" for name in CAPNO_COLS}
    values["DATE"] = date
    values["TIME"] = time
    values["RR"] = rr
    values["CO2 Wave"] = "1.0"
    values["EtCO2"] = "2.0"
    values["SpO2"] = "98"
    values["PR"] = "70"
    return "\t".join(values[name] for name in CAPNO_COLS) + "\n"


def test_shift_time_meta_attr_handles_objects_and_strings():
    meta = SimpleNamespace(capno_shift="2.5")
    t = np.array([1.0, 2.0, 3.0])
    shifted = shift_time_meta_attr(t, "capno_shift", meta=meta)
    assert np.allclose(shifted, np.array([3.5, 4.5, 5.5]))


def test_parse_capnostream_file_localizes_and_spreads_samples(tmp_path: Path):
    payload = ["header\n"] * 9
    payload.extend(
        [
            _capno_row("Mar 14, 26", "08:00:00 AM", "10"),
            _capno_row("Mar 14, 26", "08:00:00 AM", "11"),
            _capno_row("Mar 14, 26", "08:00:01 AM", "12"),
            _capno_row("Mar 14, 26", "08:00:02 AM", "13"),
            _capno_row("Mar 14, 26", "08:00:02 AM", "14"),
            _capno_row("Mar 14, 26", "08:00:02 AM", "15"),
            _capno_row("Mar 14, 26", "08:00:03 AM", "16"),
            _capno_row("Mar 14, 26", "08:00:04 AM", "17"),
            _capno_row("Mar 14, 26", "08:00:04 AM", "18"),
        ]
    )
    target = tmp_path / "capno.txt"
    target.write_text("".join(payload), encoding="utf-16")

    df = parse_capnostream_file(target, meta=SimpleNamespace(tz="US/Central"))

    assert list(df["RR"]) == [13, 14, 15]
    assert df["etime"].is_monotonic_increasing
    expected_start = pd.Timestamp("2026-03-14 08:00:02", tz="US/Central").timestamp()
    assert df["etime"].iloc[0] == expected_start
    assert df["etime"].iloc[-1] < expected_start + 1.0


def test_polar_h10_to_pandas_uses_serial_attr_and_single(monkeypatch, tmp_path: Path):
    class FakeDataSet:
        def to_pandas(self, **kwargs):
            assert kwargs["group_only"] == ["hr"]
            return {"hr": pd.DataFrame({"hr_t": [1.0], "hr": [60.0]})}

    def fake_from_file(filepath):
        assert filepath == tmp_path / "h10.sqlite"
        return {"abc123": FakeDataSet()}

    monkeypatch.setattr(
        "sciona.principal.adapters.parsers.PolarH10DataSet.from_file",
        fake_from_file,
    )

    result = polar_h10_to_pandas(
        tmp_path / "h10.sqlite",
        serial_attr="polar_serial",
        only=["hr"],
        single=True,
        meta=SimpleNamespace(polar_serial="abc123"),
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["hr_t", "hr"]
