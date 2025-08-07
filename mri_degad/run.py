#!/usr/bin/env python3
from pathlib import Path

from snakebids import bidsapp, plugins

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "snakebids.yml"
print(f"Running on ReadTheDocs? {'READTHEDOCS' in os.environ}")
print(f"Config file path: {CONFIG_PATH}")
print(f"Exists? {CONFIG_PATH.exists()}")

app = bidsapp.app(
    [
        plugins.SnakemakeBidsApp(Path(__file__).resolve().parent),
        plugins.BidsValidator(),
        plugins.Version(distribution="MRI_Degad"),
    ]
)


def get_parser():
    """Exposes parser for sphinx doc generation, cwd is the docs dir."""
    return app.build_parser().parser


if __name__ == "__main__":
    app.run()
