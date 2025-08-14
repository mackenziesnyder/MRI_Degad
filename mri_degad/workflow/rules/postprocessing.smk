rule fuse_image:
    input:
        degad_coronal = bids(
            root=work,
            datatype="degad",
            desc="degad_coronal",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        degad_axial = bids(
            root=work,
            datatype="degad",
            desc="degad_axial",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        degad_sagittal = bids(
            root=work,
            datatype="degad",
            desc="degad_sagittal",
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
        "../scripts/fuse_img.py"  

rule skull_strip_degad:
    input:
        in_img = bids(
            root=work,
            datatype="denoised",
            desc="degad",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
            )
    output:
        out_im_skull_stripped = bids(
            root=work,
            datatype="skull_stripped",
            desc="degad_skull_stripped",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    shell: 
        "mri_synthstrip -i {input.in_img} -o {output.out_im_skull_stripped}"
        

rule skull_strip_gad:
    input:
        in_img = bids(
            root=str(Path(config["bids_dir"])),
            datatype="anat",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
            ),
    output:
        out_im_skull_stripped = bids(
            root=work,
            datatype="skull_stripped",
            desc="gad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
            ),
    shell:
        "mri_synthstrip -i {input.in_img} -o {output.out_im_skull_stripped}"
        

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

rule registration_skullstripped:
    input:
        fixed_gad = bids(
            root=work,
            datatype="skull_stripped",
            desc="gad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        moving_degad = bids(
            root=work,
            datatype="skull_stripped",
            desc="degad_skull_stripped",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output:
        out_im = bids(
            root=work,
            datatype="registration",
            desc="degad_reg_skull_stripped",
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

rule extract_vasc_mask_skullstripped:
    input:
        gad_img = bids(
            root=work,
            datatype="skull_stripped",
            desc="gad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        degad_img = bids(
            root=work,
            datatype="registration",
            desc="degad_reg_skull_stripped",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output:
        out_mask = bids(
            root=work,
            datatype="mask",
            suffix="mask_skullstripped.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/extract_contrast_mask.py"