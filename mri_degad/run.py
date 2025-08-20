#!/usr/bin/env python3
import os
from pathlib import Path

from snakebids import bidsapp, plugins

if "__file__" not in globals():
    __file__ = "../mri_degad/run.py"


app = bidsapp.app(
    [
        plugins.SnakemakeBidsApp(Path(__file__).resolve().parent),
        plugins.BidsValidator(),
        plugins.Version(distribution="MRI_Degad"),
        plugins.CliConfig("parse_args"),
        plugins.ComponentEdit("pybids_inputs"),
    ]
)


def get_parser():
    """Exposes parser for sphinx doc generation, cwd is the docs dir."""
    return app.build_parser().parser


if __name__ == "__main__":
    app.run()
