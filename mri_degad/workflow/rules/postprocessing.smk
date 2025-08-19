rule fuse_degad_image:
    input:
        degad_coronal = bids(
            root=work,
            datatype="degad",
            desc="degad_coronal",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        degad_axial = bids(
            root=work,
            datatype="degad",
            desc="degad_axial",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        degad_sagittal = bids(
            root=work,
            datatype="degad",
            desc="degad_sagittal",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output: 
        degad = bids(
            root=work,
            datatype="degad",
            desc="fused",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/fuse_img.py"  

rule skull_strip_degad:
    input:
        in_img = bids(
            root=work,
            datatype="degad",
            desc="fused",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
            )
    output:
        out_im_skull_stripped = bids(
            root=work,
            datatype="skull_stripped",
            desc="degad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    resources:
        mem_mb = 8000,
        gpus=1 if config["use_gpu"] else 0,
    threads: 8
    log:
        bids(
            root="logs",
            suffix="skull_strip_degad.txt",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
    shell: 
        "mri_synthstrip -i {input.in_img} -o {output.out_im_skull_stripped} &>{log}"
        

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
    resources:
        mem_mb = 8000,
        gpus=1 if config["use_gpu"] else 0,
    threads: 8
    log:
        bids(
            root="logs",
            suffix="skull_strip_gad.txt",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
    shell:
        "mri_synthstrip -i {input.in_img} -o {output.out_im_skull_stripped} &>{log}"
        

rule register_degad_to_gad:
    input:
        fixed_im = bids(
            root=str(Path(config["bids_dir"])),
            datatype="anat",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        moving_im = bids(
            root=work,
            datatype="degad",
            desc="fused",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )  
    output:
        out_im = bids(
            root=work,
            datatype="registration",
            suffix="T1w.nii.gz",
            desc="degad_to_gad",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/registration.py"

rule register_degad_to_gad_skull_stripped:
    input:
        fixed_im = bids(
            root=work,
            datatype="skull_stripped",
            desc="gad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        moving_im = bids(
            root=work,
            datatype="skull_stripped",
            desc="degad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output:
        out_im = bids(
            root=work,
            datatype="registration",
            desc="degad_to_gad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="degad",
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
            desc="degad_to_gad",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )   
    output:
        out_mask = bids(
            root=work,
            datatype="mask",
            suffix="mask.nii.gz",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/extract_contrast_mask.py"

rule extract_vasc_mask_skull_stripped:
    input:
        gad_img = bids(
            root=work,
            datatype="skull_stripped",
            desc="gad_skull_stripped",
            acq="gad",
            suffix="T1w.nii.gz",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        degad_img = bids(
            root=work,
            datatype="registration",
            desc="degad_to_gad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output:
        out_mask = bids(
            root=work,
            datatype="mask",
            suffix="mask_skull_stripped.nii.gz",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/extract_contrast_mask.py"