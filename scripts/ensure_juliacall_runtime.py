from __future__ import annotations

from sciona.julia_runtime import prewarm_juliacall_project


def main() -> None:
    cfg = prewarm_juliacall_project()
    print(cfg.project)


if __name__ == "__main__":
    main()
