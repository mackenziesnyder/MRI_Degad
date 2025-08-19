
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

If you're developing the app and want to install development dependencies such as linters and formatters, run:

```
uv pip install -e .[dev]
```
