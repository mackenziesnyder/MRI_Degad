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

rule apply_model_cornonal:
    input:
        t1w_gad = bids(
            root=work,
            datatype="resampled",
            desc="resampled",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        model_dir = Path(download_dir) / "models"
    params:
        config_path = "/local/scratch/MRI_Degad/mri_degad/workflow/scripts/config_inference.json",
        view = "coronal"
    output: 
        degad_img = bids(
            root=work,
            datatype="degad",
            desc="degad_coronal",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script: '../scripts/model_apply.py'

rule apply_model_axial:
    input:
        t1w_gad = bids(
            root=work,
            datatype="resampled",
            desc="resampled",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        model_dir = Path(download_dir) / "models"
    params:
        config_path = "/local/scratch/MRI_Degad/mri_degad/workflow/scripts/config_inference.json",
        view = "axial"
    output: 
        degad_img = bids(
            root=work,
            datatype="degad",
            desc="degad_axial",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script: '../scripts/model_apply.py'

rule apply_model_sagittal:
    input:
        t1w_gad = bids(
            root=work,
            datatype="resampled",
            desc="resampled",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        model_dir = Path(download_dir) / "models"
    params:
        config_path = "/local/scratch/MRI_Degad/mri_degad/workflow/scripts/config_inference.json",
        view = "sagittal"
    output: 
        degad_img = bids(
            root=work,
            datatype="degad",
            desc="degad_sagittal",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script: '../scripts/model_apply.py'