
# Installation

Install from github with pip:

```bash
pip install -e git+https://github.com/mackenziesnyder/mri_degad#egg=mri_degad
```

Note: you can re-run this command to re-install with the latest version

# Running the app

Do a dry-run first (`-n`) and simply print (`-p`) what would be run:

```bash
mri_degad /path/to/bids/dir /path/to/output/dir participant -np
```

Run the app, using all cores::

```bash
mri_degad /path/to/bids/dir /path/to/output/dir participant --cores all
```

If any workflow rules require containers, then run with the `--use-singularity` option.

# Generating a report

After your processing is complete, you can use snakemake's `--report` feature to generate
an HTML report. This report will include a graph of all the jobs run, with clickable nodes
to inspect the shell command or python code used in each job, along with the config files and
run times for each job. Workflows may also contain append images for quality assurance or to
summarize outputs, by using the `report(...)` function on any snakemake output.

To generate a report, run:

```bash
mri_degad /path/to/bids/dir /path/to/output/dir participant --report
```

# Compute Canada Instructions

## Setting up a dev environment

Here are some instructions to get your python environment set-up on graham to run mri_degad:

### Create a virtualenv and activate it:

```bash
cd $SCRATCH
module load python/3
virtualenv venv_mri_degad
source venv_mri_degad/bin/activate
```

### Follow the steps above to install from github repository

## Install job submission helpers

Snakemake can submit jobs with SLURM, but you need to set-up a Snakemake profile to enable this. The Khan lab has a
snakemake profile that is configured for graham but is customizable upon install, please see `cc-slurm <https://github.com/khanlab/cc-slurm>` for more info.

If you don't need Snakemake to parallelize jobs across different nodes, you can make use of the simple job submission wrappers in `neuroglia-helpers <https://github.com/khanlab/neuroglia-helpers>`, e.g. `regularSubmit` or `regularInteractive` wrappers.

These are used in the instructions below.

## Running jobs on Compute Canada

In an interactive job (for testing):

```bash
regularInteractive -n 8 mri_degad bids_dir out_dir participant --participant_label 001 -j 8
```

Submitting a job (for larger cores, more subjects), still single job, but snakemake will parallelize over the 32 cores:

```bash
regularSubmit -j Fat mri_degad bids_dir out_dir participant  -j 32
```

Scaling up to ~hundred subjects (needs cc-slurm snakemake profile installed), submits 1 16core job per subject:

```bash
mri_degad bids_dir out_dir participant  --profile cc-slurm
```

Scaling up to even more subjects (uses group-components to bundle multiple subjects in each job), 1 32core job for N subjects (e.g. 10):

```bash
mri_degad bids_dir out_dir participant  --profile cc-slurm --group-components subj=10
```
