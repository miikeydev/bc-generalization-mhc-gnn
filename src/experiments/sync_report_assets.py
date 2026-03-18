from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="outputs/analysis/paper_figures/gcn_family/manifest.json",
        help="Path to the generated paper-figure manifest.",
    )
    parser.add_argument(
        "--report-root",
        default="docs/report",
        help="Path to the report repository root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    report_root = Path(args.report_root).resolve()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assets_root = ensure_dir(report_root / "assets" / "generated")
    figures_root = ensure_dir(assets_root / "figures")
    tables_root = ensure_dir(assets_root / "tables")

    synced: dict[str, list[str]] = {}
    for name, paths in manifest.items():
        synced_paths: list[str] = []
        for raw_path in paths:
            src = Path(raw_path).resolve()
            if "/figures/" in raw_path:
                dst = figures_root / src.name
            elif "/tables/" in raw_path:
                dst = tables_root / src.name
            else:
                dst = assets_root / src.name
            shutil.copy2(src, dst)
            synced_paths.append(str(dst))
        synced[name] = synced_paths

    write_json(assets_root / "manifest.json", synced)

    print("\nSynced report assets:")
    for name, paths in synced.items():
        print(f"- {name}")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
