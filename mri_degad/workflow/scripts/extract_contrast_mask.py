import nibabel as nib
import numpy as np
import os

def extract_contrast_mask_red_overlay(gad_img_path, degad_img_path, out_rgb_path):
    # Expand paths
    gad_img_path = os.path.expanduser(gad_img_path)
    degad_img_path = os.path.expanduser(degad_img_path)
    out_rgb_path = os.path.expanduser(out_rgb_path)

    # Load images
    gad_img = nib.load(gad_img_path)
    degad_img = nib.load(degad_img_path)

    gad_data = gad_img.get_fdata()
    degad_data = degad_img.get_fdata()

    if gad_data.shape != degad_data.shape:
        raise ValueError("Input images must have the same shape.")

    # Enhancement map
    enhancement = gad_data - degad_data

    # Threshold to extract vasculature
    threshold = np.percentile(enhancement, 99)
    vessel_mask = enhancement > threshold  # shape: (X, Y, Z)

    # Create RGB image (X, Y, Z, 3)
    rgb_shape = vessel_mask.shape + (3,)
    rgb_img = np.zeros(rgb_shape, dtype=np.uint8)

    # Red: vessel pixels
    rgb_img[..., 0] = (vessel_mask * 255).astype(np.uint8)  # Red channel
    rgb_img[..., 1] = 0  # Green
    rgb_img[..., 2] = 0  # Blue

    # Save RGB NIfTI
    rgb_nifti = nib.Nifti1Image(rgb_img, gad_img.affine)
    nib.save(rgb_nifti, out_rgb_path)

# Example usage
extract_contrast_mask_red_overlay(
    "/localscratch/out.nii.gz",
    "/cifs/khan_new/trainees/msalma29/degad_project/inference_results_v2/sub-P030/gad_recon_pred_fused.nii.gz",
    "/localscratch/red_overlay.nii.gz"
)
