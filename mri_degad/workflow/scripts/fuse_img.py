import nibabel as nib

def fuse_image(coronal, axial, sagittal, output):

    coronal_img = nib.load(coronal)
    axial_img = nib.load(axial)
    sagittal_img = nib.load(sagittal)

if __name__ == "__main__":
    fuse_image(
        snakemake.input.degad_coronal,
        snakemake.input.degad_axial,
        snakemake.input.degad_sagittal,
        snakemake.output.degad
    )