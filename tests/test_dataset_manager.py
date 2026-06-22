"""Tests for S3 dataset manager, caching, and auto-resolution in the execution runner."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from sciona.visualizer.dataset_manager import DatasetManager
from sciona.visualizer.runner import parse_input_value


def test_dataset_manager_listing():
    """Verify that DatasetManager registers default mocks and can scan files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dm = DatasetManager(cache_dir=tmp_path)

        # 1. Check default mocks
        datasets = dm.list_datasets()
        fqns = [ds["fqn"] for ds in datasets]
        assert "s3://sciona-datasets/biosppy/ecg_sample_1.npz" in fqns
        assert "s3://sciona-datasets/matrix/dense_matrix_100x100.npz" in fqns

        # 2. Check type definitions
        for ds in datasets:
            assert "name" in ds
            assert "type" in ds
            assert "shape" in ds
            assert "description" in ds


def test_dataset_manager_mock_loading():
    """Verify that loading datasets fallback to synthetic mock data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dm = DatasetManager(cache_dir=tmp_path)

        # 1. ECG signal mock
        ecg = dm.load_dataset("s3://sciona-datasets/biosppy/ecg_sample_1.npz")
        assert isinstance(ecg, np.ndarray)
        assert len(ecg) == 36000

        # 2. Dense Matrix mock
        matrix = dm.load_dataset("s3://sciona-datasets/matrix/dense_matrix_100x100.npz")
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (100, 100)


def test_execution_runner_s3_resolution():
    """Verify that parse_input_value resolves s3:// URIs into actual arrays."""
    # When parse_input_value gets an S3 FQN, it should auto-load the mock dataset
    ecg_uri = "s3://sciona-datasets/biosppy/ecg_sample_1.npz"
    val = parse_input_value(ecg_uri, "NDArray[np.float64]")
    assert isinstance(val, np.ndarray)
    assert len(val) == 36000


def test_curated_inputs_lookup():
    """Verify that get_curated_inputs_for_primitive returns list of FQNs or empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dm = DatasetManager(cache_dir=tmp_path)

        # Should load gracefully and not crash
        inputs = dm.get_curated_inputs_for_primitive("sciona.atoms.signal_processing.fft")
        assert isinstance(inputs, list)
