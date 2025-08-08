import torch
import json
import nibabel as nib
import numpy as np
import os
from torch.utils.data import DataLoader
from model_helpers.src.training.trainer_lightning import Model
from model_helpers.src.data.out_of_distribution_dataset import NiftiTestDataset, resample_to_original

def run_inference(view, data_path, checkpoint_path, config, output_path):
    
    with open(config) as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Disable loss computations during inference
    config['skip_ada_multi_losses_norm'] = True
    enable_sap = config['data_loader'].get('enable_SAP', False)

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

    # Reconstruct volume from predicted slices
    slices_sorted = sorted(pred_slices, key=lambda s: s[0])
    volume = np.stack([s[1] for s in slices_sorted], axis=0)
    volume = np.squeeze(volume)

    # Convert to original space (if needed)
    was_resampled = test_dataset.motion_resampled
    volume_original = resample_to_original(volume, test_dataset.ras_shape, was_resampled)

    # Save NIfTI if requested
    if output_path:
        nii_out = nib.Nifti1Image(volume_original.astype(np.float32), test_dataset.affine)
        nib.save(nii_out, output_path)

    return volume_original

if __name__ == "__main__":

    run_inference(
        view=snakemake.params.view,
        data_path=snakemake.input.t1w_gad,
        checkpoint_path=snakemake.input.model_dir,
        config=snakemake.params.config_path,
        output_path=snakemake.output.degad_img
    )