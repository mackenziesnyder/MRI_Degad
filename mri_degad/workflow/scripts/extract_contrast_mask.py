import nibabel as nib
import numpy as np

def extract_contrast_mask(gad_img, degad_img, out_mask):
    
    # load images
    gad_img = nib.load(gad_img)
    degad_img = nib.load(degad_img)

    gadolinium_data = gad_img.get_fdata()
    predicted_data = degad_img.get_fdata()

    # enhancement map
    enhancement = gadolinium_data - predicted_data

    # threshold to extract vasculature
    threshold = np.percentile(enhancement, 99)
    vessel_mask = enhancement > threshold

    # Save mask
    vessel_mask_nifti = nib.Nifti1Image(vessel_mask.astype(np.uint8), gad_img.affine)
    nib.save(vessel_mask_nifti, out_mask)

extract_contrast_mask(
    snakemake.input.gad_img,
    snakemake.input.degad_img,
    snakemake.output.out_mask
)