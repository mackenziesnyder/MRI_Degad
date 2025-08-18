import nibabel as nib
import numpy as np
from scipy.ndimage import median_filter

def fuse_image(coronal, axial, sagittal, output):
    
    # Load and undo permutations
    axial_img = nib.load(axial)
    coronal_img = nib.load(coronal)
    sagittal_img = nib.load(sagittal)

    axial_data = axial_img.get_fdata()
    coronal_data = coronal_img.get_fdata()
    sagittal_data = sagittal_img.get_fdata()

    Dx, Dy, Dz = axial_data.shape
    eps = 1e-3

    filter_sizes = [7]
    clip_values = [3]

    # Start grid search
    for filter_size in filter_sizes:
        for upper_clip in clip_values:
            corrected_axial = np.zeros_like(axial_data)
            corrected_sagittal = np.zeros_like(sagittal_data)
            corrected_coronal = np.zeros_like(coronal_data)

            # Axial correction
            for i in range(Dz):
                a = axial_data[:, :, i]
                b = sagittal_data[:, :, i]
                c = coronal_data[:, :, i]

                a_med = median_filter(a, size=(filter_size, 1))
                b_med = median_filter(b, size=(filter_size, 1))
                c_med = median_filter(c, size=(1, filter_size))

                r_sag = np.clip((b_med + eps) / (a_med + eps), 0, upper_clip)
                r_cor = np.clip((c_med + eps) / (a_med + eps), 0, upper_clip)

                r_comb_axial = (r_sag + r_cor) / 2
                corrected_axial[:, :, i] = a * r_comb_axial

            # Sagittal correction
            for i in range(Dx):
                a = axial_data[i, :, :]
                b = sagittal_data[i, :, :]
                c = coronal_data[i, :, :]

                b_med = median_filter(b, size=(1, filter_size))
                a_med = median_filter(a, size=(1, filter_size))
                c_med = median_filter(c, size=(filter_size, 1))

                r_ax = np.clip((a_med + eps) / (b_med + eps), 0, upper_clip)
                r_cor = np.clip((c_med + eps) / (b_med + eps), 0, upper_clip)

                r_comb_sagittal = (r_ax + r_cor) / 2
                corrected_sagittal[i, :, :] = b * r_comb_sagittal

            # Coronal correction
            for i in range(Dy):
                a = axial_data[:, i, :]
                b = sagittal_data[:, i, :]
                c = coronal_data[:, i, :]

                c_med = median_filter(c, size=(1, filter_size))
                a_med = median_filter(a, size=(1, filter_size))
                b_med = median_filter(b, size=(filter_size, 1))

                r_ax = np.clip((a_med + eps) / (c_med + eps), 0, upper_clip)
                r_sag = np.clip((b_med + eps) / (c_med + eps), 0, upper_clip)

                r_comb_coronal = (r_ax + r_sag) / 2
                corrected_coronal[:, i, :] = c * r_comb_coronal

            # Fuse results
            mean_fused = (corrected_axial + corrected_sagittal + corrected_coronal) / 3
            # median_fused = np.median(np.stack([corrected_axial, corrected_sagittal, corrected_coronal]), axis=0)

    fused_img = nib.Nifti1Image(mean_fused, affine=axial_img.affine, header=axial_img.header)
    nib.save(fused_img, output)

if __name__ == "__main__":
    fuse_image(
        snakemake.input.degad_coronal,
        snakemake.input.degad_axial,
        snakemake.input.degad_sagittal,
        snakemake.output.degad
    ) 