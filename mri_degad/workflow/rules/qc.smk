rule quality_control:
    input:
        gad_img=bids(
            root=work,
            datatype="normalize",
            desc="normalize_minmax",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
        degad_img=bids(
            root=work,
            datatype="registration",
            desc="degad_to_gad",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    output:
        out_html=bids(
            root=work,
            datatype="qc",
            suffix="qc.html",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    script:
        "../scripts/qc.py"


rule quality_control_skull_stripped:
    input:
        gad_img=bids(
            root=work,
            datatype="skull_strip",
            desc="gad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
        degad_img=bids(
            root=work,
            datatype="registration",
            desc="degad_to_gad_skull_stripped",
            suffix="T1w.nii.gz",
            acq="degad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    output:
        out_html=bids(
            root=work,
            datatype="qc",
            suffix="qc_skull_stripped.html",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"},
        ),
    script:
        "../scripts/qc.py"
