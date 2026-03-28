#!/usr/bin/env python3
"""Rebuild the curated O3 version-evolution analysis package."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).resolve().parent / "tmp_nibi" / "version_evolution_equiv_report.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
