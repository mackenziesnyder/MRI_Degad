# Output Files
In the specified `/path/to/output/dir`, there will be the following outputs:

```
/path/to/output/dir/
└── config
├── logs
├── work
├── .snakebids
└── .snakemake
```

The `config` folder, along with the hidden `.snakebids` and `.snakemake` folders
contain a record of the code and parameters used, and paths to the inputs. The `logs` folder contains .txt log files from the skull_stripping and model apply rules. 

## Work Directory 
After running the workflow, the `/path/to/output/dir` folder will contain a `work` directory. All the preprocessed nii.gz files for the mri_degad program will be in the `work` directory with the following structure:

```
work/
└── sub-{subject}
    ├── n4biascorr
    ├── degad
    ├── vasc_mask
    ├── normalize
    ├── qc
    ├── registration
    └── resample
```

If the `--skull-strip` flag is included:
```
work/
└── sub-{subject}
    ├── n4biascorr
    ├── degad
    ├── vasc_mask
    ├── normalize
    ├── qc
    ├── registration
    ├── resample
    └── skull_strip
```
