rule n4_bias_correction_nongad:
    input:
        input_im=bids(
            root=str(Path(config["bids_dir"])),
            datatype="anat",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    output:
        corrected_im=bids(
            root=work,
            datatype="n4biascorr",
            desc="n4_bias_corr",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    script:
        "../scripts/n4_bias_corr.py"


# isotropic resampling
rule isotropic_resampling_nongad:
    input:
        input_im=bids(
            root=work,
            datatype="n4biascorr",
            desc="n4_bias_corr",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    output:
        resam_im=bids(
            root=work,
            datatype="resample",
            desc="resampled",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    script:
        "../scripts/resample_img.py"


# minmax normalization
rule normalize_minmax_nongad:
    input:
        input_im=bids(
            root=work,
            datatype="resample",
            desc="resampled",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    output:
        norm_im=bids(
            root=work,
            datatype="normalize",
            desc="normalize_minmax",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    script:
        "../scripts/normalize.py"

rule skull_strip_nongad:
    input:
        in_img = bids(
            root=work,
            datatype="normalize",
            desc="normalize_minmax",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    output:
        out_im_skull_stripped = bids(
            root=work,
            datatype="skull_strip",
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

rule register_nongad_to_gad:
    input:
        fixed_im = bids(
            root=work,
            datatype="normalize",
            desc="normalize_minmax",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ), 
        moving_im =  bids(
            root=work,
            datatype="normalize",
            desc="normalize_minmax",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    output:
        out_im = bids(
            root=work,
            datatype="registration",
            desc="nongad_to_gad",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/registration.py"

rule register_degad_to_nongad:
    input:
        fixed_im = bids(
            root=work,
            datatype="normalize",
            desc="normalize_minmax",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
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
            datatype="skull_strip",
            desc="nongad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        moving_im =  bids(
            root=work,
            datatype="skull_strip",
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
            root=work,
            datatype="normalize",
            desc="normalize_minmax",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
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
        "../scripts/analysis.py" 

rule calculate_metrics_skull_stripped:
    input:
        nongad = bids(
            root=work,
            datatype="skull_strip",
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
        "../scripts/analysis.py" 

rule nongad_degad_qc:
    input:
        gad_img = bids(
            root=work,
            datatype="normalize",
            desc="normalize_minmax",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
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
            datatype="skull_strip",
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

rule synthsr:
    input: 
        gad_img = bids(
            root=work,
            datatype="normalize",
            desc="normalize_minmax",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    output:
        degad_synthsr_img = bids(
            root=work,
            datatype="degad",
            desc="synthsr",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    shell: mri_synthsr --i {input.gad_img} --o {output.degad_snythsr_img} --threads 4

rule download_original_model:
    params:
        url=config["resource_urls"][config["model_og"]],
    output:
        unzip_dir=directory(Path(download_dir) / "model_og"),
    shell:
        "wget https://{params.url} -O model.zip && "
        " unzip -q -d {output.unzip_dir} model.zip && "
        " rm model.zip"

rule_original_model_degad:
    input: 
        gad_img = bids(
            root=work,
            datatype="normalize",
            desc="normalize_minmax",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
        model_dir=Path(download_dir) / "model_og",
    output:
        degad_img = bids(
            root=work,
            datatype="degad",
            desc="original_model",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    script: 
        "../scripts/original_model_apply.py"

rule register_sythsr_to_gad:
    input:
        fixed_im = bids(
            root=work,
            datatype="normalize",
            desc="normalize_minmax",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
        moving_im =  bids(
            root=work,
            datatype="degad",
            desc="synthsr",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    output:
        out_im = bids(
            root=work,
            datatype="registration",
            desc="sythsr_to_gad",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/registration.py"

rule register_original_degad_to_gad:
    input:
        fixed_im = bids(
            root=work,
            datatype="normalize",
            desc="normalize_minmax",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
        moving_im =  bids(
            root=work,
            datatype="degad",
            desc="original_model",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    output:
        out_im = bids(
            root=work,
            datatype="registration",
            desc="original_model_to_gad",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/registration.py"

rule create_figures:
    input:
        gad_img = bids(
            root=work,
            datatype="normalize",
            desc="normalize_minmax",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ), 
        nongad_img = bids(
            root=work,
            datatype="registration",
            desc="nongad_to_gad",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        degad_img = bids(
            root=work,
            datatype="registration",
            suffix="T1w.nii.gz",
            desc="degad_to_gad",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
        original_degad_img = bids(
            root=work,
            datatype="registration",
            desc="original_model_to_gad",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        synthsr_img = bids(
            root=work,
            datatype="registration",
            desc="sythsr_to_gad",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        vasc_img = bids(
            root=work,
            datatype="vasc_mask",
            suffix="mask.nii.gz",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        )
    output: 
        figure = bids(
            root=work,
            datatype="figures",
            suffix="whole_compare.png",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        )
    script:
        "../scripts/create_figures.py" 