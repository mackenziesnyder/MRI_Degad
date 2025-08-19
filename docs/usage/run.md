# Running mri_degad on your data

This section goes over the command line options you will find most useful when
running mri_degad on your dataset, along with describing some issues you may face.

_Note: Please first refer to the simple example in the 
[Installation](../getting_started/installation.md) 
section, which goes over running mri_degad on a test dataset and the essential 
required options._

## Including / excluding subjects to process
By default, mri_degad will run on **all** subjects in the dataset. If you wish to 
run on only a subset of subjects, you can use the `--participant-label` flag:

```
--participant-label 001
```

which would only run on `sub-001`. You can add additional subjects by passing
a space-separated list to this option:

```
--participant-label 001 002
```

which would run for `sub-001` and `sub-002`.

Similarly, subjects can be excluded from processing using the 
`--exclude-participant-label` flag.

## BIDS Parsing limitations

mri_degad uses Snakebids, which makes use of pybids to parse a [BIDS-compliant
dataset](https://bids.neuroimaging.io/). In mri_degad, a requirement is that all inputs processed need to have the acqusition (`acq`) entity with the value set to **gad**. mri_degad should have
no problem parsing the following dataset:

```
PATH_TO_BIDS_DIR/
└── dataset_description.json
└── sub-001/
    └── anat/
        └── sub-001_acq-gad_T1w.nii.gz
└── sub-002/
    └── anat/
        ├── sub-002_acq-gad_T1w.nii.gz
...
```

as the path (with wildcards) will be interpreted as 
`sub-{subject}_acq-{acq}_T1w.nii.gz`.

However, because of the way Snakebids
and Snakemake operate, one limitation is that the input files in your BIDS 
dataset needs to be consistent in terms of what optional BIDS entities exist in
them. In the example below, one subject has a `ses` wildcard and one does not, resulting in an error from Snakebids. 

```
PATH_TO_BIDS_DIR/
└── dataset_description.json
└── sub-001/
    └──ses-{ses}
        └── anat/
            ├── sub-001_ses-{ses}_acq-gad_T1w.nii.gz
└── sub-002/
    └── anat/
        ├── sub-002_acq-gad_T1w.nii.gz
...
```

because two distinct paths (with wildcards) would be found for T1w images:
`sub-{subject}_ses-{ses}_acq-gad_T1w.nii.gz` and `sub-{subject}_acq-gad_T1w.nii.gz`.

There will soon be added functionality in Snakebids to filter out extra files, but for now, if your dataset has these issues, you will
need to rename or remove extraneous files.