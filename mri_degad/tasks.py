import subprocess


def quality_check():
    subprocess.run(
        "isort mri_degad/*.py -c && black mri_degad --check && snakefmt mri_degad --check",
        shell=True,
        check=True,
    )


def quality_fix():
    subprocess.run(
        "isort mri_degad/*.py && black mri_degad && snakefmt mri_degad",
        shell=True,
        check=True,
    )
