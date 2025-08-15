
# calculate MSE, MAE, PSNR, SSIM

rule skull_strip_nogad:
    input:
        in_img = bids(
            root=str(Path(config["bids_dir"])),
            datatype="anat",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
    output:
        out_im_skull_stripped = bids(
            root=work,
            datatype="skull_stripped",
            desc="nongad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
            ),
    shell:
        "mri_synthstrip -i {input.in_img} -o {output.out_im_skull_stripped}"

rule register_degad_and_nogad_whole_image:
    input:
        fixed_gad = bids(
            root=str(Path(config["bids_dir"])),
            datatype="anat",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        moving_degad =  bids(
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
            desc="degad_reg_nogad",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/registration.py"

rule register_degad_and_nogad_ss:
    input:
        fixed_gad = bids(
            root=work,
            datatype="skull_stripped",
            desc="nongad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        moving_degad =  bids(
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
            desc="degad_reg_nogad_ss",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/registration.py"

rule calculate_metrics:
    input:
        nongad = bids(
            root=str(Path(config["bids_dir"])),
            datatype="anat",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        degad = bids(
            root=work,
            datatype="registration",
            desc="degad_reg_nogad",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output:
        metrics = bids(
            root=work,
            datatype="analysis",
            suffix="metrics.txt",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/calculate_metrics.py" 

rule calculate_metrics_ss:
    input:
        nongad = bids(
            root=work,
            datatype="skull_stripped",
            desc="nongad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        degad = bids(
            root=work,
            datatype="registration",
            desc="degad_reg_nogad_ss",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output:
        metrics = bids(
            root=work,
            datatype="analysis",
            suffix="metrics_ss.txt",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/calculate_metrics.py" 