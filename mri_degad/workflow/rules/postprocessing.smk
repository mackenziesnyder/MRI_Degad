rule image_denoising:
    input:
        degad = bids(
            root=work,
            datatype="degad",
            desc="degad",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output: 
        degad = bids(
            root=work,
            datatype="denoised",
            desc="degad",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/denoise_img.py"  

rule registration:
    input:
        fixed_gad = bids(
            root=str(Path(config["bids_dir"])),
            datatype="anat",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        moving_degad = bids(
            root=work,
            datatype="denoised",
            desc="degad",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )     
    output:
        out_im = bids(
            root=work,
            datatype="registration",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/registration.py"

rule extract_vasc_mask:
    input:
        gad_img = bids(
            root=str(Path(config["bids_dir"])),
            datatype="anat",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        degad_img = bids(
            root=work,
            datatype="registration",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )    
    output:
        out_mask = bids(
            root=work,
            datatype="mask",
            suffix="mask.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/extract_contrast_mask.py"