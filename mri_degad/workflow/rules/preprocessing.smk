# bias correction for gad files
rule n4_bias_correction:
    input:
        im = bids(
            root=str(Path(config["bids_dir"])),
            datatype="anat",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )     
    output:
        corrected_im = bids(
            root=work,
            datatype="bias_correction",
            desc="n4_bias_corr",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/n4_bias_corr.py"

# isotropic resampling
rule isotropic_resampling:
    input:
        input_im = bids(
            root=work,
            datatype="bias_correction",
            desc="n4_bias_corr",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ) 
    params:
        res=resolution
    output:
        resam_im = bids(
            root=work,
            datatype="resampled",
            desc="resampled",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ) 
    script:
        "../scripts/resample_img.py"
