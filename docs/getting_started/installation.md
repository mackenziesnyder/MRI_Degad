
# Installation

To install the app, first install `uv`. On macOS and Linux, run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, run:
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternatively, you can install it from PyPI using:

```
pip install uv
```

or with pipx:

```
pipx install uv
```

Once uv is installed, create and activate a virtual environment:

```
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Then install the package in editable mode:

```
uv pip install -e .
```

# Running the app

To verify that the installation was successful, run:

```
mri_degad -h
```

This should display the help message for the CLI. If it doesn’t, ensure that your virtual environment is activated and that the installation completed without errors.

To use the BIDS App, run:

```
mri_degad /path/to/bids/dataset /path/to/output/derivatives participant --cores all
```

Replace /path/to/bids/dataset with the path to your BIDS-compliant input dataset and /path/to/output/derivatives with the desired output directory.

If you're developing the app and want to install development dependencies such as linters and formatters, please refer to the instructions in the [Contributing](../contributing/contributing.md) page.

# Dry-run / testing your workflow

Using Snakemake\'s dry-run option (`--dry-run`/`-n`) is an easy way to verify
any changes made to the workflow are working direcctly. The `tests/data` folder 
contains a _fake_ BIDS dataset (i.e. dataset with zero-sized files) that is 
useful for verifying different aspects of the workflow. These dry-run tests are 
part of the automated Github actions that are run for every commit.

```
mri_degad tests/data /test/data/derivatives participant --cores all -n
```

This performs the baseline test, in which a user
may use mri_degad.