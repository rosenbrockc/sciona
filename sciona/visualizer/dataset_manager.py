"""S3 dataset caching, manifest management, and loading utilities."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)

# Default cache directory in the workspace
WORKSPACE_DIR = Path("/Users/conrad/personal/sciona-matcher")
CACHE_DIR = WORKSPACE_DIR / ".sciona_datasets_cache"



def load_curated_inputs_from_repos() -> dict[str, list[str]]:
    """Scans all sibling repos for cdg.json files and extracts curated_inputs mappings."""
    from sciona.sdk import _resolve_default_repos

    mapping = {}
    repos = _resolve_default_repos()
    for repo in repos:
        for cdg_path in repo.rglob("cdg.json"):
            if "solution_cdgs" in str(cdg_path):
                continue
            try:
                with open(cdg_path, "r") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    continue
            except Exception:
                continue

            parts = cdg_path.parts
            if "src" in parts:
                src_idx = parts.index("src")
                module_parts = parts[src_idx + 1 : -1]
                module_prefix = ".".join(module_parts)
            else:
                module_prefix = ""

            cdg_inputs = data.get("curated_inputs", [])

            for node in data.get("nodes", []):
                node_name = node.get("name", "")
                if not node_name:
                    continue
                node_inputs = node.get("curated_inputs", [])
                combined_inputs = list(dict.fromkeys(cdg_inputs + node_inputs))
                if not combined_inputs:
                    continue

                if module_prefix:
                    fqdn = f"{module_prefix}.{node_name}"
                    mapping[fqdn] = combined_inputs

                mapping[node_name] = combined_inputs

    return mapping


class DatasetManager:
    """Manages downloading, caching, loading, and mocking S3-backed canonical datasets."""

    def __init__(self, cache_dir: Path = CACHE_DIR, s3_bucket: str = "sciona-datasets"):
        self.cache_dir = Path(cache_dir)
        self.s3_bucket = s3_bucket
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._curated_mappings: dict[str, list[str]] | None = None

    def get_curated_inputs_for_primitive(self, primitive_name: str) -> list[str]:
        """Looks up the curated datasets suggested for a given primitive name."""
        if self._curated_mappings is None:
            try:
                self._curated_mappings = load_curated_inputs_from_repos()
            except Exception as e:
                logger.error(f"Error loading curated inputs from repos: {e}")
                self._curated_mappings = {}

        if primitive_name in self._curated_mappings:
            return self._curated_mappings[primitive_name]

        short_name = primitive_name.rsplit(".", 1)[-1]
        if short_name in self._curated_mappings:
            return self._curated_mappings[short_name]

        return []

    def list_datasets(self) -> list[dict[str, Any]]:
        """Lists all known curated datasets, scanning cache and adding default mocks."""
        datasets = {}

        # 1. Start with standard built-in mock datasets
        default_fqns = [
            "s3://sciona-datasets/biosppy/ecg_sample_1.npz",
            "s3://sciona-datasets/matrix/dense_matrix_100x100.npz",
            "s3://sciona-datasets/signal/sinusoid_50hz.npz",
        ]
        for fqn in default_fqns:
            datasets[fqn] = self._generate_mock_manifest(fqn)

        # 2. Scan local cache directory for any other manifest files (.json)
        for path in self.cache_dir.rglob("*.json"):
            if path.name == "run_metadata.json":
                continue
            try:
                with open(path, "r") as f:
                    manifest = json.load(f)
                    if "fqn" in manifest:
                        datasets[manifest["fqn"]] = manifest
            except Exception:
                pass

        return list(datasets.values())

    def _parse_s3_uri(self, uri: str) -> tuple[str, str]:
        """Parses s3://bucket/key -> (bucket, key)"""
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {uri}")
        parts = uri[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    def get_dataset_path(self, fqn: str) -> Path:
        """Returns the local cached path for a dataset FQN."""
        _, key = self._parse_s3_uri(fqn)
        return self.cache_dir / key

    def get_manifest_path(self, fqn: str) -> Path:
        """Returns the local cached path for a dataset's manifest."""
        return Path(str(self.get_dataset_path(fqn)) + ".json")

    def download_dataset(self, fqn: str) -> bool:
        """Downloads the data file and manifest file from S3 to local cache."""
        try:
            import boto3  # type: ignore[import-untyped]
            from botocore.exceptions import ClientError, NoCredentialsError  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("boto3 not installed, cannot download from S3. Using mock/cached data.")
            return False

        bucket, key = self._parse_s3_uri(fqn)
        local_data_path = self.get_dataset_path(fqn)
        local_manifest_path = self.get_manifest_path(fqn)

        local_data_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            session = boto3.Session(profile_name="sciona")
        except Exception:
            session = boto3.Session()
        s3_client = session.client("s3")
        try:
            # Download manifest first
            logger.info(f"Downloading manifest from S3: {fqn}.json")
            s3_client.download_file(bucket, f"{key}.json", str(local_manifest_path))

            # Download data file
            logger.info(f"Downloading data from S3: {fqn}")
            s3_client.download_file(bucket, key, str(local_data_path))
            return True
        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"S3 download failed for {fqn}: {e}. Falling back to mock/local.")
            return False

    def load_dataset(self, fqn: str) -> Any:
        """Loads and returns the cached/downloaded dataset, falling back to mock if needed."""
        local_path = self.get_dataset_path(fqn)

        # Try to download if not cached
        if not local_path.exists():
            success = self.download_dataset(fqn)
            if not success:
                return self._generate_mock_dataset(fqn)

        # File exists, load it
        suffix = local_path.suffix.lower()
        try:
            if suffix == ".npz":
                with np.load(local_path, allow_pickle=True) as data:
                    keys = list(data.keys())
                    if len(keys) == 1:
                        return data[keys[0]]
                    return {k: data[k] for k in keys}
            elif suffix == ".npy":
                return np.load(local_path, allow_pickle=True)
            elif suffix in (".pkl", ".pickle"):
                with open(local_path, "rb") as f:
                    return pickle.load(f)
            elif suffix == ".json":
                with open(local_path, "r") as f:
                    return json.load(f)
            else:
                raise ValueError(f"Unsupported dataset format: {suffix}")
        except Exception as e:
            logger.error(f"Error loading cached dataset {local_path}: {e}. Generating mock fallback.")
            return self._generate_mock_dataset(fqn)

    def load_manifest(self, fqn: str) -> Dict[str, Any]:
        """Loads and returns the manifest dictionary, falling back to mock if needed."""
        local_path = self.get_manifest_path(fqn)
        if not local_path.exists():
            self.download_dataset(fqn)

        if local_path.exists():
            try:
                with open(local_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass

        return self._generate_mock_manifest(fqn)

    def _generate_mock_dataset(self, fqn: str) -> Any:
        """Generates synthetic mock datasets for offline/tutorial use."""
        logger.info(f"Generating mock dataset for FQN: {fqn}")
        _, key = self._parse_s3_uri(fqn)
        name = Path(key).name.lower()

        if "ecg" in name:
            # Synthetic 1D ECG raw signal
            t = np.linspace(0, 10, 36000)
            sig = 0.5 * np.sin(2 * np.pi * 0.1 * t)  # respiration baseline
            for peak_t in range(1, 10):
                sig += np.exp(-((t - peak_t) / 0.05) ** 2) * 1.5  # R-peak
            sig += 0.05 * np.random.randn(len(t))
            return sig
        elif "matrix" in name or "dense" in name:
            return np.random.randn(100, 100)
        elif "sinusoid" in name:
            t = np.linspace(0, 1, 1000)
            return np.sin(2 * np.pi * 50 * t)
        else:
            return np.random.randn(100)

    def _generate_mock_manifest(self, fqn: str) -> Dict[str, Any]:
        """Generates a mock manifest matching the generated mock dataset."""
        _, key = self._parse_s3_uri(fqn)
        filename = Path(key).name
        name = filename.replace("_", " ").title()

        if "ecg" in filename:
            shape = [36000, 1]
            dtype = "float64"
            desc = "Mock raw ECG sample data generated locally."
        elif "matrix" in filename or "dense" in filename:
            shape = [100, 100]
            dtype = "float64"
            desc = "Mock 100x100 random float matrix."
        else:
            shape = [100]
            dtype = "float64"
            desc = f"Mock dataset for {filename}."

        return {
            "fqn": fqn,
            "name": name,
            "type": "numpy.ndarray",
            "shape": shape,
            "dtype": dtype,
            "description": desc,
            "attribution": {
                "source": "AGEO-Matcher Local Mock Data Provider",
                "url": "https://github.com/rosenbrockc/sciona",
                "license": "MIT",
            },
        }
