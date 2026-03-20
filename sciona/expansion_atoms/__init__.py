"""Runtime atoms and registries for expansion rule domains.

Each domain provides:
  - A ``runtime_*.py`` module with pure, deterministic atom functions
  - A ``*_registry.py`` module with declaration metadata (import path,
    type signature, docstring)

Domains:
  - sequential_filter — Kalman/particle filter diagnostics
  - signal_event_rate — signal→event→rate pipeline pre/post-processing
  - mcmc — HMC sampler diagnostics and adaptive corrections
  - graph_traversal — structural pre-checks and traversal monitoring
"""
