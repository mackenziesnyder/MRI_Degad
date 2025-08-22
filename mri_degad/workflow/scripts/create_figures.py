import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def plot_nifti_grid(nifti_files, titles, output):
    """
    Plot axial, sagittal, and coronal middle slices of multiple NIfTI images
    in a 3xN grid (rows=orientations, cols=images).
    """
    n_imgs = len(nifti_files)
    fig, axes = plt.subplots(3, n_imgs, figsize=(3*n_imgs, 9),
                             gridspec_kw={"hspace": 0.02, "wspace": 0.05})  # 👈 uniform spacing

    for col, f in enumerate(nifti_files):
        img = nib.load(f).get_fdata()
        x, y, z = img.shape

        # Middle indices
        mid_x, mid_y, mid_z = x // 2, y // 2, z // 2

        # Axial (Z slice)
        axes[0, col].imshow(np.rot90(img[:, :, mid_z]), cmap="gray")
        axes[0, col].axis("off")
        if col == 0:
            axes[0, col].set_ylabel("Axial", fontsize=14)

        # Sagittal (X slice)
        axes[1, col].imshow(np.rot90(img[mid_x, :, :]), cmap="gray")
        axes[1, col].axis("off")
        if col == 0:
            axes[1, col].set_ylabel("Sagittal", fontsize=14)

        # Coronal (Y slice)
        axes[2, col].imshow(np.rot90(img[:, mid_y, :]), cmap="gray")
        axes[2, col].axis("off")
        if col == 0:
            axes[2, col].set_ylabel("Coronal", fontsize=14)

        # Column titles = provided titles
        axes[0, col].set_title(titles[col], fontsize=16)

    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {output}")


if __name__ == "__main__":
    nifti_files = [
        snakemake.input.gad_img,
        snakemake.input.nongad_img,
        snakemake.input.degad_img,
        snakemake.input.original_degad_img,
        snakemake.input.synthsr_img,
        snakemake.input.vasc_img,
    ]

    titles = [
        "Gad",
        "Nongad",
        "Degad",
        "Original Degad Model",
        "Synthsr",
        "Vascular mask",
    ]

    plot_nifti_grid(nifti_files, titles, snakemake.output.figure)