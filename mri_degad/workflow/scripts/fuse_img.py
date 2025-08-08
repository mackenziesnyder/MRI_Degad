import nibabel as nib
import numpy as np
from scipy.signal import medfilt2d

# adapted from Wenyao's 2 volume merge code 
def fuse_image(coronal, axial, sagittal, output):

    axial_img = nib.load(axial)
    coronal_img = nib.load(coronal)
    sagittal_img = nib.load(sagittal)

    axial_data = axial_img.get_fdata()
    coronal_data = coronal_img.get_fdata()
    sagittal_data = sagittal_img.get_fdata()

    Dx, Dy, Dz = axial_data.shape
    
    fused_volume = np.zeros_like(axial_data)

    for i in range(Dz):
        
        # take the i-th slice in the axial orientation
        a = axial_data[:, :, i]
        c = coronal_data[:, :, i]
        s = sagittal_data[:, :, i]

        # median filter each
        a1 = medfilt2d(a, kernel_size=(11, 1))
        c1 = medfilt2d(c, kernel_size=(11, 1))
        s1 = medfilt2d(s, kernel_size=(11, 1))

        # compute ratios (avoiding div-by-zero)
        r_ac = (c1 + 0.001) / (a1 + 0.001)
        r_as = (s1 + 0.001) / (a1 + 0.001)

        # clip extreme values
        r_ac = np.clip(r_ac, None, 2)
        r_as = np.clip(r_as, None, 2)

        # adjust axial slice with both ratios and average
        adjusted_a = a * ((r_ac + r_as) / 2)

        # combine into fused volume (you could also average a,c,s)
        fused_volume[:, :, i] = adjusted_a

    # Save fused volume
    fused_img = nib.Nifti1Image(fused_volume, affine=axial_img.affine, header=axial_img.header)
    nib.save(fused_img, output)


if __name__ == "__main__":
    fuse_image(
        snakemake.input.degad_coronal,
        snakemake.input.degad_axial,
        snakemake.input.degad_sagittal,
        snakemake.output.degad
    )