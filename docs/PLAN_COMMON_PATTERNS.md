# Implementation Plan: `common_patterns` Atom Metadata

## Context

Partial matches account for 34% of CDG stage evaluations. The retrieval
system finds one correct atom but doesn't know it needs companions. The
`common_patterns` field lets atoms declare "when you use me, you usually
also need these" — enabling the architect to expand a single retrieval hit
into a multi-atom binding.

## Schema

Add `common_patterns` to atomic nodes in cdg.json:

```json
{
  "node_id": "gp_train_cholesky",
  "common_patterns": [
    {
      "pattern_id": "gaussian_process_interpolation",
      "description": "Fit GP from training data, predict mean and variance at new points",
      "atoms": [
        "gp_train_cholesky",
        "gp_posterior_mean",
        "gp_posterior_variance"
      ],
      "ordering": "sequential",
      "when": "Stage describes GP fitting, interpolation, or kriging"
    }
  ]
}
```

### Field definitions

- `pattern_id`: Unique identifier for this composition pattern. Snake_case.
  Must be consistent across all atoms that participate in the pattern.
- `description`: What the composed pipeline does as a whole.
- `atoms`: Ordered list of atom node_ids that form the pattern. The atom
  declaring the pattern MUST be in this list. Use bare node_id (not FQDN)
  — the validation script resolves existence across all repos.
- `ordering`: How atoms relate:
  - `"sequential"` — atoms execute in listed order, output of one feeds next
  - `"parallel"` — atoms execute independently, results are combined
  - `"alternatives"` — atoms are interchangeable choices for the same role
- `when`: 1-sentence hint for the architect: under what CDG stage description
  should this pattern be activated.

### Invariants (enforced by validation)

1. **Self-inclusion**: Every atom that declares a pattern must include itself
   in the `atoms` list.
2. **Symmetry**: If atom A declares pattern P referencing atom B, then atom B
   must also declare pattern P referencing atom A (with the same `atoms` list).
3. **Existence**: Every atom referenced in any pattern must exist in some
   cdg.json across all repos.
4. **Consistency**: All atoms in a pattern must declare the same `pattern_id`
   with the same `atoms` list (order may differ for `parallel`/`alternatives`
   but must match for `sequential`).
5. **No orphan patterns**: A pattern with only one atom is invalid (use
   aliases instead).

## Examples

### Sklearn fit/transform triad

```json
// In variance_threshold cdg.json
{
  "node_id": "variance_threshold_fit",
  "common_patterns": [
    {
      "pattern_id": "variance_threshold_pipeline",
      "description": "Fit variance threshold, compute support mask, filter features",
      "atoms": ["variance_threshold_fit", "variance_threshold_support_mask", "variance_threshold_transform"],
      "ordering": "sequential",
      "when": "Stage describes removing low-variance features"
    }
  ]
}
```

All three atoms (`variance_threshold_fit`, `variance_threshold_support_mask`,
`variance_threshold_transform`) declare the same pattern.

### Audio mel spectrogram composition

```json
// In mel_filterbank cdg.json
{
  "node_id": "mel_filterbank",
  "common_patterns": [
    {
      "pattern_id": "mel_spectrogram_pipeline",
      "description": "Build mel filterbank, project power spectrum, apply log scaling",
      "atoms": ["mel_filterbank", "apply_mel_filterbank", "log_mel_spectrogram"],
      "ordering": "sequential",
      "when": "Stage describes computing mel spectrogram from audio"
    }
  ]
}
```

### Ensembling alternatives

```json
// In voting_classifier_soft_probabilities cdg.json
{
  "node_id": "voting_classifier_soft_probabilities",
  "common_patterns": [
    {
      "pattern_id": "prediction_averaging",
      "description": "Average predictions from multiple models",
      "atoms": ["voting_classifier_soft_probabilities", "voting_regressor_average", "ranked_prediction_blend", "fold_ensemble_average"],
      "ordering": "alternatives",
      "when": "Stage describes ensembling, blending, or averaging model predictions"
    }
  ]
}
```

### EKF sensor fusion pipeline

```json
// In update_step cdg.json
{
  "node_id": "update_step",
  "common_patterns": [
    {
      "pattern_id": "kalman_filter_cycle",
      "description": "Predict state forward, then fuse measurement via Kalman update",
      "atoms": ["predict_step", "update_step"],
      "ordering": "sequential",
      "when": "Stage describes Kalman filter, EKF, or sensor fusion"
    },
    {
      "pattern_id": "kalman_filter_with_smoothing",
      "description": "Full EKF cycle with RTS backward smoother",
      "atoms": ["predict_step", "update_step", "rts_smooth"],
      "ordering": "sequential",
      "when": "Stage describes EKF with smoothing or forward-backward filtering"
    }
  ]
}
```

### GP interpolation

```json
{
  "node_id": "gp_train_cholesky",
  "common_patterns": [
    {
      "pattern_id": "gaussian_process_interpolation",
      "description": "Fit GP, predict mean and variance at new points",
      "atoms": ["gp_train_cholesky", "gp_posterior_mean", "gp_posterior_variance"],
      "ordering": "sequential",
      "when": "Stage describes GP fitting, interpolation, or kriging"
    }
  ]
}
```

## Changes to existing code

### 1. Model: `AlgorithmicPrimitive` (models.py)

```python
class CommonPattern(BaseModel):
    pattern_id: str
    description: str
    atoms: list[str]
    ordering: str = "sequential"  # sequential, parallel, alternatives
    when: str = ""

class AlgorithmicPrimitive(BaseModel):
    # ... existing fields ...
    aliases: list[str] = Field(default_factory=list)
    common_patterns: list[CommonPattern] = Field(default_factory=list)
```

### 2. Catalog: pattern index (catalog.py)

Build a pattern index during `add()`, similar to aliases:

```python
class PrimitiveCatalog:
    def __init__(self):
        # ... existing ...
        self._patterns: dict[str, list[str]] = {}  # pattern_id -> [atom_names]

    def add(self, primitive):
        # ... existing alias registration ...
        for pattern in primitive.common_patterns:
            self._patterns.setdefault(pattern.pattern_id, [])
            if primitive.name not in self._patterns[pattern.pattern_id]:
                self._patterns[pattern.pattern_id].append(primitive.name)

    def get_pattern_companions(self, atom_name: str) -> list[list[str]]:
        """Return all patterns this atom participates in."""
        prim = self.get(atom_name)
        if prim is None:
            return []
        return [p.atoms for p in prim.common_patterns]
```

### 3. Retrieval expansion in `find_matching_primitives()`

After retrieving top-k, expand results with pattern companions:

```python
def find_matching_primitives_with_patterns(
    self, node: AlgorithmicNode, k: int = 5
) -> tuple[list[AlgorithmicPrimitive], list[CommonPattern]]:
    """Find primitives and suggest composition patterns."""
    prims = self.find_matching_primitives(node, k=k)
    suggested_patterns = []
    for prim in prims:
        for pattern in prim.common_patterns:
            if pattern not in suggested_patterns:
                suggested_patterns.append(pattern)
    return prims, suggested_patterns
```

### 4. CDG loader (test + runtime)

Read `common_patterns` from cdg.json nodes (same as `aliases`):

```python
AlgorithmicPrimitive(
    # ... existing ...
    aliases=node.get("aliases", []),
    common_patterns=[
        CommonPattern(**p) for p in node.get("common_patterns", [])
    ],
)
```

### 5. API: expose patterns in search results (sdk.py)

```python
@dataclass
class AtomMatch:
    atom_name: str
    atom_fqdn: str
    score: float
    category: str
    description: str
    suggested_patterns: list[dict] = field(default_factory=list)
```

## Validation script

### `scripts/validate_common_patterns.py`

```python
#!/usr/bin/env python3
"""Validate common_patterns consistency across all atom repos.

Checks:
1. Self-inclusion: declaring atom is in its own pattern's atoms list
2. Symmetry: if A references B in pattern P, B must also declare P referencing A
3. Existence: every atom in every pattern exists in some cdg.json
4. Consistency: all atoms in pattern P declare the same atoms list
5. No orphans: patterns with < 2 atoms are invalid
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    repos = [
        Path.home() / "personal" / name
        for name in [
            "sciona-atoms", "sciona-atoms-ml", "sciona-atoms-dl",
            "sciona-atoms-bio", "sciona-atoms-physics", "sciona-atoms-signal",
            "sciona-atoms-cs", "sciona-atoms-geo", "sciona-atoms-fintech",
            "sciona-atoms-robotics",
        ]
    ]

    # Phase 1: collect all atoms and their patterns
    all_atoms: set[str] = set()
    # pattern_id -> {declaring_atom -> sorted atoms list}
    pattern_declarations: dict[str, dict[str, list[str]]] = defaultdict(dict)
    errors: list[str] = []

    for repo in repos:
        if not repo.exists():
            continue
        for cdg_path in repo.rglob("cdg.json"):
            if "solution_cdgs" in str(cdg_path):
                continue
            try:
                data = json.loads(cdg_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            for node in data.get("nodes", []):
                if node.get("status") != "atomic":
                    continue
                atom_name = node["node_id"]
                all_atoms.add(atom_name)

                for pattern in node.get("common_patterns", []):
                    pid = pattern["pattern_id"]
                    atoms_list = pattern["atoms"]

                    # Check 1: self-inclusion
                    if atom_name not in atoms_list:
                        errors.append(
                            f"SELF_INCLUSION: {atom_name} declares pattern "
                            f"'{pid}' but is not in its atoms list: {atoms_list}"
                        )

                    # Check 5: no orphans
                    if len(atoms_list) < 2:
                        errors.append(
                            f"ORPHAN: {atom_name} declares pattern '{pid}' "
                            f"with only {len(atoms_list)} atom(s) — use aliases instead"
                        )

                    pattern_declarations[pid][atom_name] = sorted(atoms_list)

    # Phase 2: cross-atom checks
    for pid, declarations in pattern_declarations.items():
        all_pattern_atoms = set()
        for atoms_list in declarations.values():
            all_pattern_atoms.update(atoms_list)

        # Check 3: existence
        for atom in all_pattern_atoms:
            if atom not in all_atoms:
                errors.append(
                    f"EXISTENCE: pattern '{pid}' references atom '{atom}' "
                    f"which does not exist in any cdg.json"
                )

        # Check 2: symmetry
        for atom in all_pattern_atoms:
            if atom not in declarations:
                declared_by = list(declarations.keys())
                errors.append(
                    f"SYMMETRY: atom '{atom}' is referenced in pattern "
                    f"'{pid}' (declared by {declared_by}) but does not "
                    f"declare the pattern itself"
                )

        # Check 4: consistency
        canonical = None
        for declaring_atom, atoms_list in declarations.items():
            if canonical is None:
                canonical = atoms_list
            elif atoms_list != canonical:
                errors.append(
                    f"CONSISTENCY: pattern '{pid}' has inconsistent atoms "
                    f"lists: {declaring_atom} declares {atoms_list} but "
                    f"expected {canonical}"
                )

    # Report
    total_patterns = len(pattern_declarations)
    total_atoms_with_patterns = len({
        atom for decls in pattern_declarations.values() for atom in decls
    })

    print(f"Scanned {len(all_atoms)} atoms across {len(repos)} repos")
    print(f"Found {total_patterns} patterns declared by {total_atoms_with_patterns} atoms")
    print()

    if errors:
        print(f"ERRORS ({len(errors)}):")
        for err in sorted(errors):
            print(f"  {err}")
        sys.exit(1)
    else:
        print("All pattern validations passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
```

### Integration with CI

Add to the test suite:

```python
# tests/test_common_patterns.py
def test_common_patterns_valid():
    """Run the pattern validation script and assert no errors."""
    result = subprocess.run(
        [sys.executable, "scripts/validate_common_patterns.py"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
```

## Files to create/modify

| File | Action |
|------|--------|
| `sciona/architect/models.py` | Add `CommonPattern` model + field on `AlgorithmicPrimitive` |
| `sciona/architect/catalog.py` | Add `_patterns` dict, populate in `add()`, add `get_pattern_companions()` |
| `sciona/sdk.py` | Expose patterns in `AtomMatch` results |
| `sciona/api_models.py` | Add `suggested_patterns` field to `AtomMatch` |
| `tests/test_retrieval_solution_cdgs.py` | Read `common_patterns` from cdg.json |
| `scripts/validate_common_patterns.py` | **Create** — validation script |
| `tests/test_common_patterns.py` | **Create** — CI integration test |

## Rollout plan

### Phase 1: Infrastructure (this PR)
- Add `CommonPattern` model + catalog support + validation script
- No atom changes yet — field defaults to empty list

### Phase 2: High-impact patterns (~20 pattern declarations)
Annotate the atoms that appear most often as partial matches:

| Pattern ID | Atoms | Partial matches resolved |
|-----------|-------|------------------------|
| `variance_threshold_pipeline` | fit, support_mask, transform | 2 stages |
| `mel_spectrogram_pipeline` | mel_filterbank, apply_mel_filterbank, log_mel_spectrogram | 3 stages |
| `kalman_filter_cycle` | predict_step, update_step | 2 stages |
| `kalman_filter_with_smoothing` | predict_step, update_step, rts_smooth | 1 stage |
| `gaussian_process_interpolation` | gp_train_cholesky, gp_posterior_mean, gp_posterior_variance | 2 stages |
| `prediction_averaging` | voting_soft, voting_reg, ranked_blend, fold_average | 5 stages |
| `image_patch_pipeline` | extract_patches_2d, reconstruct_from_patches_2d | 3 stages |
| `tfidf_pipeline` | tfidf_vectorizer_fit, tfidf_transform | 3 stages |
| `pca_embedding_pipeline` | pca_fit, pca_whiten_reduce, l2_normalize | 2 stages |
| `cross_validation_pipeline` | cross_validation, fold_ensemble_average | 3 stages |
| `bio_tagging_pipeline` | tokenize, bio_decode, char_to_token_offsets | 3 stages |
| `stacking_pipeline` | cross_validation, stacking_meta_feature_matrix | 2 stages |
| `nms_detection_pipeline` | nms, threshold_detections | 2 stages |
| `morphological_cleanup` | morphological_close, fill_holes, filter_components_by_area | 2 stages |
| `pseudo_labeling_pipeline` | extract_pseudo_labels, self_training_fit | 2 stages |
| `macenko_stain_pipeline` | macenko_stain_vectors, macenko_normalize | 1 stage |
| `sigmoid_calibration_pipeline` | sigmoid_calibration_fit, sigmoid_calibration_predict | 2 stages |
| `pdr_navigation` | detect_steps, estimate_step_length, pdr_position_update | 1 stage |
| `feature_hashing_pipeline` | feature_hasher_csr_matrix, hashing_vectorizer_transform | 1 stage |
| `als_recommendation` | als_user_update, als_item_update, cooccurrence_candidates | 1 stage |

### Phase 3: Architect integration
Update the architect's stage-binding logic to:
1. Retrieve top-k atoms for a stage
2. Check if any have `common_patterns`
3. If a pattern's `when` clause matches the stage description, propose
   the full pattern as a multi-atom binding
4. Score the pattern match higher than individual atom matches

This turns partial matches into correct matches without creating new atoms.
