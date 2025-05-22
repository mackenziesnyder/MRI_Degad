# Contributing to MRI-Degad

MRI_Degad python package uses Poetry pacakge manager to manage its dependencies. You’ll need it installed on your machine before contributing to the software. Installation instructions can be found on the 
[Poetry website](https://python-poetry.org/docs/master/#installation).

MRI_Degad currently only caters to T1w modality images.

Note: These instructions are only recommended if you are making changes to the MRI-Degad codebase and committing these back to the repository or if you are using Snakemake’s cluster execution profiles.

## Setup the development environment

Once Poetry is available, clone this repository and install all dependencies (including `dev`):

```
git clone https://github.com/mackenziesnyder/MRI_DeGad.git
cd MRI_Degad 
poetry install --with dev 
```

Poetry will automatically create a virtual environment. To customize where 
these virtual environments are stored, see the poetry docs 
[here](https://python-poetry.org/docs/configuration/).

Then, you can run autoafids:

```
mri_degad -h
```

or you can activate a virtual environment shell and run mri_degad directly:

```
poetry shell
mri_degad
```

You can exit the poetry shell with `exit`

## Running and fixing code format quality

MRI_Degad uses [poethepoet](https://github.com/nat-n/poethepoet) as a task runner.
You can see what commands are available by running:

```
poetry run poe 
```

We use a few tools, including `ruff`, `snakefmt`, and `yamlfix` to ensure 
formatting and style of our codebase is consistent. There are two task runners 
you can use to check and fix your code, which can be invoked with:

```
poetry run poe quality-check
poetry run poe quality
```

_Note: If you are in a poetry shell, you do not need to prepend `poetry run` to
the command._

## Dry-run / testing your workflow

Using Snakemake\'s dry-run option (`--dry-run`/`-n`) is an easy way to verify
any changes made to the workflow are working direcctly. The `tests/data` folder 
contains a _fake_ BIDS dataset (i.e. dataset with zero-sized files) that is 
useful for verifying different aspects of the workflow. These dry-run tests are 
part of the automated Github actions that are run for every commit.

You can invoke the pre-configured task via 
[poethepoet](https://github.com/nat-n/poethepoet) to perform a dry-run:

```
poetry run poe test_base
```

This performs a number of tests, involving different scenarios in which a user
may use MRI-Degad.

## Questions, Issues, Suggestions, and Other Feedback
Please reach out if you have any questions, suggestions, or other feedback related to this software—either through email (msnyde26@uwo.ca) or the discussions page.