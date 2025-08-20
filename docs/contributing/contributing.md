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
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
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

We use a few tools, including `black`, `snakefmt`, and `isort` to ensure 
formatting and style of our codebase is consistent. There are two task runners 
you can use to check and fix your code, which can be invoked with:

```
uv run quality-fix
uv run quality-check
```

## Questions, Issues, Suggestions, and Other Feedback
Please reach out if you have any questions, suggestions, or other feedback related to this software—either through email (m25snyde@uwaterloo.ca) or the discussions page.