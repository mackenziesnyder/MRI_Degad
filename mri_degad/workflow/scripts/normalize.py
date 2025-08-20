import nibabel as nib


def normalize(input_im, output_im):
    """Normalize MRI image"""

    nii = nib.load(input_im)
    nii_affine = nii.affine
    nii_data = nii.get_fdata()
    nii_data_normalized = (nii_data - nii_data.min()) / (
        nii_data.max() - nii_data.min()
    )
    nib.save(nib.Nifti1Image(nii_data_normalized, affine=nii_affine), output_im)


if __name__ == "__main__":
    normalize(
        input_im=snakemake.input["input_im"],
        output_im=snakemake.output["norm_im"],
    )
