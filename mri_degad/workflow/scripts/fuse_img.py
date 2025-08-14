import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.signal import medfilt2d

def undo_permutation(volume, view):
    if view == 'axial':
        return np.transpose(volume, (1, 2, 0))  # inverse of (2,0,1)
    elif view == 'coronal':
        return np.transpose(volume, (1, 0, 2))  # inverse of (1,0,2)
    elif view == 'sagittal':
        return volume  # no permutation
    else:
        raise ValueError(f"Unknown view: {view}")

def fuse_image(coronal, axial, sagittal, output):
    # Load and undo permutations
    axial_img = nib.load(axial)
    coronal_img = nib.load(coronal)
    sagittal_img = nib.load(sagittal)

    print(f"axial shape: {axial_img.shape}")
    print(f"coronal shape: {coronal_img.shape}")
    print(f"sagittal shape: {sagittal_img.shape}")

    axial_data = undo_permutation(axial_img.get_fdata(), 'axial')
    coronal_data = undo_permutation(coronal_img.get_fdata(), 'coronal')
    sagittal_data = undo_permutation(sagittal_img.get_fdata(), 'sagittal')

    print(f"axial shape: {axial_img.shape}")
    print(f"coronal shape: {coronal_img.shape}")
    print(f"sagittal shape: {sagittal_img.shape}")
    
    # Wrap back into NIfTI objects (keep affine from axial as target space)
    axial_img_fixed = nib.Nifti1Image(axial_data, affine=axial_img.affine)
    coronal_img_fixed = nib.Nifti1Image(coronal_data, affine=coronal_img.affine)
    sagittal_img_fixed = nib.Nifti1Image(sagittal_data, affine=sagittal_img.affine)

    # Resample coronal & sagittal to match axial voxel grid
    coronal_resampled = resample_from_to(coronal_img_fixed, axial_img_fixed)
    sagittal_resampled = resample_from_to(sagittal_img_fixed, axial_img_fixed)

    axial_data = axial_img_fixed.get_fdata()
    coronal_data = coronal_resampled.get_fdata()
    sagittal_data = sagittal_resampled.get_fdata()

    Dx, Dy, Dz = axial_data.shape
    fused_volume = np.zeros_like(axial_data)

    for i in range(Dz):
        a = axial_data[:, :, i]
        c = coronal_data[:, :, i]
        s = sagittal_data[:, :, i]

        a1 = medfilt2d(a, kernel_size=(11, 1))
        c1 = medfilt2d(c, kernel_size=(11, 1))
        s1 = medfilt2d(s, kernel_size=(11, 1))

        r_ac = (c1 + 0.001) / (a1 + 0.001)
        r_as = (s1 + 0.001) / (a1 + 0.001)

        r_ac = np.clip(r_ac, None, 2)
        r_as = np.clip(r_as, None, 2)

        adjusted_a = a * ((r_ac + r_as) / 2)
        fused_volume[:, :, i] = adjusted_a

    fused_img = nib.Nifti1Image(fused_volume, affine=axial_img.affine, header=axial_img.header)
    nib.save(fused_img, output)
if __name__ == "__main__":
    fuse_image(
        snakemake.input.degad_coronal,
        snakemake.input.degad_axial,
        snakemake.input.degad_sagittal,
        snakemake.output.degad
    ) #the wrong axis try and fuse