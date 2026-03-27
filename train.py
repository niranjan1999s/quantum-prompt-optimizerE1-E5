from __future__ import annotations

import pathlib
import sys


def _add_src_to_path() -> None:
    root = pathlib.Path(__file__).resolve().parent
    src = root / "src"
    sys.path.insert(0, str(src))


def main() -> None:
    _add_src_to_path()
    from training.trainer import main as trainer_main

    trainer_main()


if __name__ == "__main__":
    main()

