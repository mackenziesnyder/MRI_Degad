# rules that download the model and apply the model to the inputs

rule download_cnn_model:
    params:
        url=config["resource_urls"][config["model"]],
    output:
        unzip_dir=directory(Path(download_dir) / "models")
    shell:
        "wget https://{params.url} -O model.zip && "
        " unzip -q -d {output.unzip_dir} model.zip && "
        " rm model.zip"

rule apply_model_coronal:
    input:
        t1w_gad = bids(
            root=work,
            datatype="normalize",
            desc="normalize_minmax",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        model_dir = Path(download_dir) / "models"
    params:
        config_path = str(Path(workflow.basedir) / "scripts" / "model_helpers" / "config_inference.json"),
        view = "coronal"
    resources:
        gpus=1 if config["use_gpu"] else 0,
    output: 
        degad_img = bids(
            root=work,
            datatype="degad",
            desc="degad_coronal",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script: '../scripts/model_apply.py'

rule apply_model_axial:
    input:
        t1w_gad = bids(
            root=work,
            datatype="normalize",
            desc="normalize_minmax",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        model_dir = Path(download_dir) / "models"
    params:
        config_path = str(Path(workflow.basedir) / "scripts" / "model_helpers" / "config_inference.json"),
        view = "axial"
    resources:
        gpus=1 if config["use_gpu"] else 0,
    output: 
        degad_img = bids(
            root=work,
            datatype="degad",
            desc="degad_axial",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script: '../scripts/model_apply.py'

rule apply_model_sagittal:
    input:
        t1w_gad = bids(
            root=work,
            datatype="normalize",
            desc="normalize_minmax",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        model_dir = Path(download_dir) / "models"
    params:
        config_path = str(Path(workflow.basedir) / "scripts" / "model_helpers" / "config_inference.json"),
        view = "sagittal"
    resources:
        gpus=1 if config["use_gpu"] else 0,
    output: 
        degad_img = bids(
            root=work,
            datatype="degad",
            desc="degad_sagittal",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script: '../scripts/model_apply.py'