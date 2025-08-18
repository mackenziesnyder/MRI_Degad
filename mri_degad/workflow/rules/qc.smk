rule quality_control:
    input:
        degad_img = bids(
            root=work,
            datatype="registration",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        gad_img = bids(
            root=work,
            datatype="resampled",
            desc="resampled",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
    output:
        out_html = bids(
            root=work,
            datatype="qc",
            suffix="qc.html",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    script:
        "../scripts/qc.py" 