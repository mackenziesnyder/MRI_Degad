
# calculate MAE, PSNR, SSIM

rule skull_strip_nongad:
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
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
    resources:
        mem_mb = 8000,
        gpus=1 if config["use_gpu"] else 0,
    threads: 8
    log:
        bids(
            root="logs",
            suffix="skull_strip_nongad.txt",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
    shell:
        "mri_synthstrip -i {input.in_img} -o {output.out_im_skull_stripped} &>{log}"

rule register_degad_to_nongad:
    input:
        fixed_im = bids(
            root=str(Path(config["bids_dir"])),
            datatype="anat",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        moving_im =  bids(
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
            desc="degad_to_nongad",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/registration.py"

rule register_degad_to_nongad_skull_stripped:
    input:
        fixed_im = bids(
            root=work,
            datatype="skull_stripped",
            desc="nongad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        moving_im =  bids(
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
            desc="degad_to_nongad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="degad",
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
            desc="degad_to_nongad",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output:
        metrics = bids(
            root=work,
            datatype="analysis",
            suffix="metrics.txt",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/calculate_metrics.py" 

rule calculate_metrics_skull_stripped:
    input:
        nongad = bids(
            root=work,
            datatype="skull_stripped",
            desc="nongad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        degad = bids(
            root=work,
            datatype="registration",
            desc="degad_to_nongad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output:
        metrics = bids(
            root=work,
            datatype="analysis",
            suffix="metrics_skull_stripped.txt",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/calculate_metrics.py" 

rule nongad_degad_qc:
    input:
        gad_img = bids(
            root=str(Path(config["bids_dir"])),
            datatype="anat",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        degad_img = bids(
            root=work,
            datatype="registration",
            desc="degad_to_gad",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
    output:
        out_html = bids(
            root=work,
            datatype="qc",
            suffix="qc_nongad_to_degad.html",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/qc.py" 

rule nongad_degad_skull_stripped_qc:
    input:
        gad_img = bids(
            root=work,
            datatype="skull_stripped",
            desc="nongad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        degad_img = bids(
            root=work,
            datatype="registration",
            desc="degad_to_nongad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output:
        out_html = bids(
            root=work,
            datatype="qc",
            suffix="qc_nongad_to_degad_skull_stripped.html",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/qc.py" 