from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.paper_figures import generate_paper_figure_assets
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis/paper_figures_gcn_family.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    manifest = generate_paper_figure_assets(config=config, config_path=config_path)
    print("\nGenerated paper-figure assets:")
    for name, paths in manifest.items():
        print(f"- {name}")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
