# Example: ECG Heart Rate Detection

This guide demonstrates how to use **AGEO-Matcher** to develop a verified, optimal algorithm for detecting heart rate from raw ECG signals from scratch.

## 1. Finding an Optimal CDG Structure

The **Architect** (Round 1) decomposes a high-level goal into a **Conceptual Dependency Graph (CDG)**. To find the *optimal* structure for a specific metric (e.g., precision), we use the **Principal** (Meta-Optimizer).

### Initial Decomposition
First, generate an initial decomposition of the goal:

```bash
ageom decompose "Detect heart rate from raw ECG signal" 
  --output ecg_initial.json 
  --max-depth 3
```

### NAS-style Optimization
Use the `optimize` command to run a Neural Architecture Search (NAS) style loop. This will explore different decomposition strategies (e.g., varying the granularity of filtering or peak detection) to minimize a loss function on a benchmark dataset.

```bash
# optimize for precision over 20 trials using a standard JSON dataset
ageom optimize "Detect heart rate from raw ECG signal" 
  --benchmark data/ecg_bench.json 
  --metric precision 
  --trials 20 
  --output ecg_optimal.json

# OR: optimize using a templated adapter dataset (adapter.yml)
ageom optimize "Detect heart rate from raw ECG signal" 
  --benchmark data/ecg_adapter.yml 
  --metric precision 
  --trials 20
```

The Principal uses **Optuna** for HPO and **Ghost Simulation** to prune invalid architectures early, eventually saving the best-performing CDG to `ecg_optimal.json`.

---

## 2. Creating an Executable

Once you have an optimal CDG, you need to ground its abstract nodes into actual executable code using the **Hunter** (Round 2) and **Synthesizer** (Round 3).

### Grounding and Assembly
Run the full orchestration to match CDG nodes to verified library functions (e.g., from `numpy`, `scipy`, or `ageoa`) and assemble them into a Python script:

```bash
ageom run "Detect heart rate from raw ECG signal" 
  --prover python 
  --output ./build/ecg_pipeline/
```

This command performs:
1. **Matching**: Finds verified atoms for nodes like "Bandpass Filter" or "R-Peak Detection".
2. **Synthesis**: Assembles the matches into a compilable Python file (`ecg_pipeline_verified.py`).
3. **Repair**: If there are type or shape mismatches, the Synthesizer uses an LLM-driven repair loop to fix the "glue" code.

### Exporting the Package
Export the verified source into a clean, distributable Python package with FFI bindings if necessary:

```bash
ageom export ./build/ecg_pipeline/ecg_verified.py 
  --target python-pkg 
  --output-dir ./dist/ecg_processor
```

---

## 3. Validation with a Templated Dataset

Finally, validate that your generated executable performs as expected on real-world data.

### Profiling Performance
Use the `profile` command to evaluate the exported artifact against your benchmark dataset. This provides empirical telemetry (latency, memory, precision) for the entire pipeline.

```bash
ageom profile 
  --cdg ./build/ecg_pipeline/cdg.json 
  --artifact ./dist/ecg_processor/main.py 
  --dataset data/ecg_test_set.json 
  --metric precision
```

### Templated Dataset (`data/ecg_adapter.yml`)
For complex sensor data spanning multiple files or groups, use a templated adapter:

```yaml
name: ECG_Study
meta:
  source: metadata.csv
  reader:
    fqn: pandas.read_csv
  user: user_id
  device:
    - name: sensor_id
      description: hardware serial
groups:
  ecg:
    source: "user_$(user_id)/ecg.csv"
    columns: [timestamp, lead_ii]
  accelerometer:
    source: "user_$(user_id)/accel.csv"
    columns: [timestamp, x, y, z]
```

When using an `adapter.yml`, AGEO-Matcher automatically:
1. Resolves file paths using the `varset`.
2. Loads and merges sensor groups into a unified manifest.
3. Produces a `dataset_manifest.json` for the executable to consume.

---

## Summary of Commands

1. **Optimize**: `ageom optimize "Goal" --benchmark data.yml`
2. **Build**: `ageom run "Goal" --prover python --output ./build/`
3. **Export**: `ageom export ./build/main.py --target python-pkg`
4. **Validate**: `ageom profile --artifact ./dist/main.py --dataset data.json`
