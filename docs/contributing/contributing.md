# Contributing to MRI-Degad

MRI_Degad python package uses uv pacakge manager to manage its dependencies. You’ll need it installed on your machine before contributing to the software. Installation instructions can be found on the 
[Installation Page](../getting_started/installation).

MRI_Degad currently only caters to T1w gadolinium-enhanced modality images.

Note: These instructions are only recommended if you are making changes to the MRI-Degad codebase and committing these back to the repository or if you are using Snakemake’s cluster execution profiles.

## Setup the development environment

Once uv is available, clone this repository and install all dependencies (including `dev`):

```
git clone https://github.com/mackenziesnyder/MRI_DeGad.git
cd MRI_Degad 
uv venv
source ./venv/Scripts/activate
uv pip install -e .[dev]
```

Then, you can run mri_degad:

```
mri_degad -h
```

You can exit the virtual environment with:

```
deactivate
```

## Running and fixing code format quality

We use a few tools, including `ruff`, `snakefmt`, and `yamlfix` to ensure 
formatting and style of our codebase is consistent. There are two task runners 
you can use to check and fix your code, which can be invoked with:

```
uv run poe quality-check
uv run poe quality
```

## Dry-run / testing your workflow

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

## Questions, Issues, Suggestions, and Other Feedback
Please reach out if you have any questions, suggestions, or other feedback related to this software—either through email (m25snyde@uwaterloo.ca) or the discussions page.