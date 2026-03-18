# Principal: Meta-Optimisation Loop

Wraps the four-round pipeline in a NAS-style optimisation loop that uses
the Architect's checkpoint time-travel for coordinate descent over decomposition
structure.

```mermaid
sequenceDiagram
    participant User
    participant Principal
    participant Optuna
    participant Architect
    participant GhostSim as Ghost Simulator
    participant Synthesizer as Synthesize Fn
    participant Sandbox as Execution Sandbox
    participant CreditAssign as Credit Assigner

    User->>Principal: optimize(goal, benchmark, metric, trials)

    Principal->>Principal: Initialise PrincipalState<br/>(goal, metric, max_trials)

    rect rgb(139, 92, 246, 0.08)
        Note over Principal,CreditAssign: Optimisation Loop (up to max_trials)

        Note over Principal: seed_population

        Principal->>Principal: current_trial += 1
        Principal->>Principal: Generate fresh thread_id

        Principal->>Architect: decompose(goal, thread_id)
        Architect-->>Principal: CDGExport

        Note over Principal: execute_forward

        Principal->>GhostSim: run_ghost_simulation(cdg, matches)
        GhostSim-->>Principal: GhostSimReport

        Principal->>Optuna: check_early_prune(ghost_report)

        alt Ghost sim failed or inf/NaN error bounds
            Optuna-->>Principal: TrialPrunedEarly
            Note over Principal: Skip to time_travel_update<br/>(error = "pruned")
        else Ghost sim OK
            Principal->>Synthesizer: synthesize_fn(cdg, matches)
            Synthesizer-->>Principal: ExportBundle (instrumented)
        end

        Note over Principal: evaluate_run

        Principal->>Sandbox: evaluate(bundle, dataset, metric)

        Note over Sandbox: Launch subprocess:<br/>python artifact.py dataset.json

        Sandbox->>Sandbox: Parse trace.jsonl<br/>-> NodeTelemetry per node

        Sandbox->>Sandbox: _compute_loss(telemetry, metric)

        alt LATENCY / FLOP_COUNT
            Sandbox->>Sandbox: sum(execution_time_ms)
        else MEMORY
            Sandbox->>Sandbox: max(peak_memory_bytes)
        else PRECISION
            Sandbox->>Sandbox: Parse MSE from stdout
        end

        Sandbox-->>Principal: BenchmarkResult(global_loss,<br/>node_telemetry)

        Principal->>Principal: Track best_loss,<br/>append to trial_history

        Note over Principal: compute_gradients

        Principal->>CreditAssign: compute_gradients(cdg,<br/>benchmark, ghost_report, metric)

        alt LATENCY / FLOP_COUNT
            CreditAssign->>CreditAssign: Per-node % of total time
        else MEMORY
            CreditAssign->>CreditAssign: Per-node % of total peak memory
        else PRECISION
            CreditAssign->>CreditAssign: Ghost-sim interval gradients (primary)<br/>+ telemetry error_expansion (fallback)
        end

        CreditAssign-->>Principal: NodeGradient[] sorted desc

        alt No gradients
            Note over Principal: done = true -> END
        else Gradients found
            Principal->>Principal: bottleneck = gradients[0]
        end

        Note over Principal: time_travel_update

        Principal->>Architect: get_state_history(thread_id)
        Architect-->>Principal: [checkpoint_entries...]

        Principal->>Principal: Find checkpoint before<br/>bottleneck node was created

        Principal->>Architect: fork(thread_id, checkpoint_id)
        Architect-->>Principal: new_thread_id

        Principal->>Architect: Inject CONSTRAINT into forked state<br/>("Previous decomposition caused bottleneck:<br/>{reason}. Re-decompose more efficiently.")

        Principal->>Architect: decompose(goal, new_thread_id)
        Architect-->>Principal: New CDGExport

        Note over Principal: route_after_update

        alt trials < max_trials and not done
            Note over Principal: Loop back to execute_forward
        else Budget exhausted or done
            Note over Principal: END
        end
    end

    Principal-->>User: Best trial artifact +<br/>trial_history
```

## State

| Field | Type | Description |
|-------|------|-------------|
| `goal` | `str` | High-level goal string |
| `metric` | `OptimizationMetric` | LATENCY, MEMORY, PRECISION, or FLOP_COUNT |
| `dataset_path` | `str` | Path to benchmark dataset |
| `max_trials` | `int` | Trial budget (default 50) |
| `current_trial` | `int` | Current trial counter |
| `best_loss` | `float` | Best global loss seen so far |
| `thread_id` | `str` | Current Architect thread ID |
| `cdg` | `CDGExport` | Current decomposition graph |
| `export_bundle` | `ExportBundle` | Current synthesised artifact |
| `ghost_report` | `GhostSimReport` | Ghost simulation result |
| `benchmark` | `BenchmarkResult` | Per-node telemetry + global loss |
| `top_gradient` | `NodeGradient` | Top bottleneck gradient |
| `bottleneck_node_id` | `str` | Node ID of the bottleneck |
| `bottleneck_reason` | `str` | Human-readable explanation |
| `trial_history` | `list[dict]` | Trial number, loss, thread_id per trial |

## Routing logic

| Router | Condition | Destination |
|--------|-----------|-------------|
| `route_after_forward` | `done = true` | END |
| | `error` (pruned early) | time_travel |
| | otherwise | evaluate |
| `route_after_gradients` | `done = true` | END |
| | `current_trial >= max_trials` | END |
| | non-pruned error | END |
| | otherwise | time_travel |
| `route_after_update` | `done = true` | END |
| | `current_trial >= max_trials` | END |
| | otherwise | forward |

## Early pruning criteria

| Condition | Source | Result |
|-----------|--------|--------|
| Ghost simulation ran and failed | `GhostSimReport.passed = false` | `TrialPrunedEarly` |
| Infinite precision gradient | `precision_gradients[nid] = inf` | `TrialPrunedEarly` |
| NaN precision gradient | `precision_gradients[nid] = NaN` | `TrialPrunedEarly` |

## Credit assignment by metric

| Metric | Signal | Formula |
|--------|--------|---------|
| LATENCY | `execution_time_ms` | `node_ms / total_ms * 100%` |
| FLOP_COUNT | `execution_time_ms` (proxy) | `node_ms / total_ms * 100%` |
| MEMORY | `peak_memory_bytes` | `node_bytes / total_bytes * 100%` |
| PRECISION | Ghost-sim interval widths (primary) or telemetry `error_expansion` (fallback) | `abs(node_grad) / total_grad * 100%` |
