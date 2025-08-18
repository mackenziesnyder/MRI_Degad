import nibabel as nib
import numpy as np
from skimage.metrics import (
    structural_similarity,
    peak_signal_noise_ratio
)

def calculate_metrics(gad_image, degad_image, output_path):
    
    # load images
    gad_img = nib.load(gad_image)
    degad_img = nib.load(degad_image)

    gad_data = gad_img.get_fdata()
    degad_data = degad_img.get_fdata()

    # ensure same shape from registration
    if gad_data.shape != degad_data.shape:
        raise ValueError(f"Image shapes do not match: {gad_data.shape} vs {degad_data.shape}")

    # compute metrics
    mae_val = np.mean(np.abs(gad_data - degad_data))
    ssim_val = structural_similarity(gad_data, degad_data, data_range=degad_data.max() - degad_data.min())
    psnr_val = peak_signal_noise_ratio(gad_data, degad_data, data_range=degad_data.max() - degad_data.min())

    # save to output file
    with open(output_path, "w") as f:
        f.write(f"MAE: {mae_val}\n")
        f.write(f"SSIM: {ssim_val}\n")
        f.write(f"PSNR: {psnr_val}\n")

if __name__ == "__main__":
    calculate_metrics(
        snakemake.input.nongad,
        snakemake.input.degad,
        snakemake.output.metrics
    )