import torch
import json
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader
from model_helpers.model import Model
from model_helpers.data import NiftiTestDataset, resample_to_original


def reconstruct_and_save(slice_dict, affine, view):
    """Reconstructs and saves a volume from slices. Pads/crops each dimension to 256 if needed."""
    if not slice_dict:
        print(f"  [No slices found, skipping.")
        return

    slices_sorted = sorted(slice_dict, key=lambda item: item[0])
    stack = np.stack([s[1] for s in slices_sorted], axis=0)
    stack = np.squeeze(stack)

    # Pad/crop each dimension to 256 if needed
    new_shape = list(stack.shape)
    pad_width = []
    for i in range(3):
        if new_shape[i] < 256:
            pad_before = (256 - new_shape[i]) // 2
            pad_after = 256 - new_shape[i] - pad_before
            pad_width.append((pad_before, pad_after))
        else:
            pad_width.append((0, 0))
    stack = np.pad(stack, pad_width, mode="constant")
    # Crop if any dimension is greater than 256
    stack = stack[0:256, 0:256, 0:256]

    stack = stack.astype(np.float32)

    if view == "coronal":
        stack = np.transpose(stack, (1, 0, 2))
    elif view == "axial":
        stack = np.transpose(stack, (1, 2, 0))
    nii_out = nib.Nifti1Image(stack, affine)
    return nii_out


def run_inference(view, data_path, checkpoint_path, config, output_path):

    with open(config) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Disable loss computations during inference
    config["skip_ada_multi_losses_norm"] = True
    enable_sap = config["data_loader"].get("enable_SAP", False)

    checkpoint_path = checkpoint_path + "/" + "last.ckpt"
    print("checkpoint_path: ", checkpoint_path)

    # Load model
    model = Model.load_from_checkpoint(checkpoint_path, config=config, device=device)
    model.eval().to(device)

    # Load test data
    test_dataset = NiftiTestDataset(view, data_path, enable_sap)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    pred_slices = []
    for i, (x, _) in enumerate(test_loader):
        with torch.no_grad():
            if enable_sap:
                x_tensors = [t.to(device) for t in x]
                pred = model.model(*x_tensors)
            else:
                x_tensor = x.to(device)
                pred = model.model(x_tensor)

        pred_slices.append((i, pred.squeeze().cpu().numpy()))

    pred_nii = reconstruct_and_save(pred_slices, test_dataset.affine, view)
    pred_recon = pred_nii.get_fdata()
    pred_recon_original = resample_to_original(pred_recon, test_dataset.ras_shape)

    # Save NIfTI if requested
    if output_path:
        nii_out = nib.Nifti1Image(
            pred_recon_original.astype(np.float32), test_dataset.affine
        )
        nib.save(nii_out, output_path)


if __name__ == "__main__":

    run_inference(
        view=snakemake.params.view,
        data_path=snakemake.input.t1w_gad,
        checkpoint_path=snakemake.input.model_dir,
        config=snakemake.params.config_path,
        output_path=snakemake.output.degad_img,
    )
