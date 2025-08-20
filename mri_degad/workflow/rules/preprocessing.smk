# bias correction for gad files
rule n4_bias_correction:
    input:
        input_im=bids(
            root=str(Path(config["bids_dir"])),
            datatype="anat",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    output:
        corrected_im=bids(
            root=work,
            datatype="n4biascorr",
            desc="n4_bias_corr",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    script:
        "../scripts/n4_bias_corr.py"


# isotropic resampling
rule isotropic_resampling:
    input:
        input_im=bids(
            root=work,
            datatype="n4biascorr",
            desc="n4_bias_corr",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    output:
        resam_im=bids(
            root=work,
            datatype="resample",
            desc="resampled",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    script:
        "../scripts/resample_img.py"


# minmax normalization
rule normalize_minmax:
    input:
        input_im=bids(
            root=work,
            datatype="resample",
            desc="resampled",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    output:
        norm_im=bids(
            root=work,
            datatype="normalize",
            desc="normalize_minmax",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    script:
        "../scripts/normalize.py"
