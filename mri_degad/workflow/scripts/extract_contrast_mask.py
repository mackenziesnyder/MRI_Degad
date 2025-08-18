import nibabel as nib
import numpy as np
import os

def extract_contrast_mask_red_overlay(gad_img_path, degad_img_path, out_prob_path):

    # Load images
    gad_img = nib.load(gad_img_path)
    degad_img = nib.load(degad_img_path)

    gad_data = gad_img.get_fdata()
    degad_data = degad_img.get_fdata()

    if gad_data.shape != degad_data.shape:
        print(f"gad shape: {gad_data.shape} is not the same as degad shape: {degad_data.shape}")
    else:
        # subtract values
        enhancement = gad_data - degad_data

        # threshold to extract vasculature
        lower, upper = np.percentile(enhancement, [98, 99.9])
        prob_map = np.clip((enhancement - lower) / (upper - lower), 0, 1)

        prob_nifti = nib.Nifti1Image(prob_map.astype(np.float32), gad_img.affine)
        nib.save(prob_nifti, out_prob_path)

extract_contrast_mask_red_overlay(
    gad_img_path=snakemake.input.gad_img, 
    degad_img_path=snakemake.input.degad_img, 
    out_prob_path=snakemake.output.out_mask
)
