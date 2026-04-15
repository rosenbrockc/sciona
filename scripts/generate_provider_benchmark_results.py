from __future__ import annotations

import json

from sciona.benchmarks.provider_results import write_provider_benchmark_results


def main() -> int:
    print(json.dumps(write_provider_benchmark_results(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
