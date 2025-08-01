# import sys
# import os
# import argparse
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import nibabel as nib
# from torch.utils.data import DataLoader
# from src.training.trainer_lightning import Model
# from src.data.out_of_distribution_dataset import NiftiTestDataset, _get_nifti_path, resample_to_original
# from src.utils.metrics import psnr, ssim_score, ms_ssim_score
# import json
# import torchio as tio
# import csv
# import gzip
# import shutil

# def reconstruct_and_save(subject_id, motion_idx, slice_dict, original_shape, affine, view, output_dir, filename_suffix):
#     """Reconstructs and saves a volume from slices. Pads/crops each dimension to 256 if needed."""
#     if not slice_dict:
#         print(f"  [Recon: {filename_suffix}] No slices found, skipping.")
#         return

#     slices_sorted = sorted(slice_dict, key=lambda item: item[0])
#     stack = np.stack([s[1] for s in slices_sorted], axis=0)
#     stack = np.squeeze(stack)

#     # Pad/crop each dimension to 256 if needed
#     new_shape = list(stack.shape)
#     pad_width = []
#     for i in range(3):
#         if new_shape[i] < 256:
#             pad_before = (256 - new_shape[i]) // 2
#             pad_after = 256 - new_shape[i] - pad_before
#             pad_width.append((pad_before, pad_after))
#         else:
#             pad_width.append((0, 0))
#     stack = np.pad(stack, pad_width, mode='constant')
#     # Crop if any dimension is greater than 256
#     stack = stack[0:256, 0:256, 0:256]

#     stack = stack.astype(np.float32)

#     if view == 'coronal':
#         stack = np.transpose(stack, (1, 0, 2))
#     elif view == 'axial':
#         stack = np.transpose(stack, (1, 2, 0))
#     nii_out = nib.Nifti1Image(stack, affine)
#     return nii_out

# def _pad_and_crop_volume(vol, target_shape, crop_shape=(256, 256, 256)):
#     # Pad to target_shape (should be the GT/original shape)
#     pad_needed = [(0, max(0, target_shape[i] - vol.shape[i])) for i in range(3)]
#     vol = torch.nn.functional.pad(vol, (0, pad_needed[2][1], 0, pad_needed[1][1], 0, pad_needed[0][1]))
#     # Center crop to crop_shape (model input)
#     start = [(vol.shape[i] - crop_shape[i]) // 2 for i in range(3)]
#     end = [start[i] + crop_shape[i] for i in range(3)]
#     vol = vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
#     return vol

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run inference on MRI motion correction model and reconstruct volumes.")
#     parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
#     parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
#     parser.add_argument('--output_dir', type=str, required=True, help='Directory to save plots and reconstructed volumes')
#     parser.add_argument('--data_path', type=str, help='Override data path from config file')
#     parser.add_argument('--subjects', type=str, nargs='+', help='List of specific subject IDs to test (overrides config subjects)')
#     parser.add_argument('--generate_plots', action='store_true', help='Whether to generate and save comparison plots')
#     parser.add_argument('--double_inference', action='store_true', help='Enable double inference (output of first inference is input to second)')
#     parser.add_argument('--quad_inference', action='store_true', help='Enable quadruple inference (output of previous inference is input to next, 4x total)')
#     parser.add_argument('--mix_inference', action='store_true', help='Enable mix-view inference: run all three views and fuse by median')
#     args = parser.parse_args()

#     # --- Setup ---
#     with open(args.config) as f:
#         config = json.load(f)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Temporarily modify config to skip adaptive loss computation for testing
#     config_for_testing = config.copy()
#     config_for_testing['skip_ada_multi_losses_norm'] = True
    
#     model = Model.load_from_checkpoint(args.checkpoint, config=config_for_testing, device=device)
#     model.eval()
#     model.to(device)
#     os.makedirs(args.output_dir, exist_ok=True)

#     # Robust data_path selection: prefer CLI, then config['data_path'], then config['dataset']
#     data_path = args.data_path or config.get('data_path') or config.get('dataset')
#     if data_path is None:
#         raise ValueError("No data path specified in config or command line (expected 'data_path' or 'dataset' in config).")

#     # Get view and enable_SAP from config, raise error if missing
#     try:
#         view = config['data_loader']['view'].lower()
#         enable_sap = config['data_loader']['enable_SAP']
#     except KeyError as e:
#         raise KeyError(f"Missing key in config['data_loader']: {e}")

#     # If mix_inference is enabled, override view handling
#     if args.mix_inference or view == 'mix':
#         mix_views = ['axial', 'sagittal', 'coronal']
#     else:
#         # If view is not valid, default to 'axial' and warn
#         if view not in ['axial', 'sagittal', 'coronal']:
#             print(f"[WARNING] Invalid view '{view}' in config. Defaulting to 'axial'.")
#             mix_views = ['axial']
#         else:
#             mix_views = [view]

#     # Determine subjects to process
#     if args.subjects:
#         # Use command line specified subjects
#         subjects = sorted(args.subjects)
#         print(f"--- Running Inference on Specified Subjects ---")
#         print(f"Command line subjects: {subjects}")
#     else:
#         # Use subjects from config or discover from data directory
#         config_subjects = config.get('subjects', [])
#         if config_subjects:
#             subjects = sorted(config_subjects)
#             print(f"--- Running Inference on Config Subjects ---")
#             print(f"Config subjects: {subjects}")
#         else:
#             # Discover subjects from data directory
#             subjects = sorted([d for d in os.listdir(data_path) if d.startswith('sub') and os.path.isdir(os.path.join(data_path, d))])
#             print(f"--- Running Inference on Discovered Subjects ---")
#             print(f"Discovered subjects: {subjects}")

#     if not subjects:
#         print(f"Error: No subjects found in data path: {data_path}")
#         print("Please specify subjects using --subjects argument or ensure data_path contains subject directories")
#         sys.exit(1)

#     print(f"Data path: {data_path}")
#     print(f"Total subjects to process: {len(subjects)}")
#     print(f"View: {view}, SAP Enabled: {enable_sap}")

#     # Initialize metrics collection
#     all_metrics = []

#     # --- Main Processing Loop ---
#     for subject_id in subjects:
#         subject_output_dir = os.path.join(args.output_dir, subject_id)
#         os.makedirs(subject_output_dir, exist_ok=True)
        
#         # For DEGAD: we only have one GAD image per subject (no multiple motion indices)
#         gad_idx = 1  # Use 1 as the default index for GAD
        
#         if len(mix_views) > 1:
#             print(f"\n--- Running MIX inference for Subject: {subject_id} ---")
#             pred_volumes = []
#             for v in mix_views:
#                 print(f"  - Inference for view: {v}")
#                 try:
#                     test_dataset = NiftiTestDataset(subject_id, gad_idx, v, data_path, enable_sap)
#                     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#                 except FileNotFoundError as e:
#                     print(f"Skipping {subject_id} view {v}: {e}")
#                     pred_volumes.append(None)
#                     continue
#                 pred_slices = []
#                 for i, (x, y) in enumerate(test_loader):
#                     if enable_sap:
#                         # DataLoader already provides correct format, just move to device
#                         x_tensors = [t.to(device) for t in x]
#                         with torch.no_grad():
#                             pred = model.model(*x_tensors)
#                     else:
#                         # DataLoader already provides correct format, just move to device
#                         x_tensor = x.to(device)
#                         with torch.no_grad():
#                             pred = model.model(x_tensor)
#                     pred_slices.append((i, pred.squeeze().cpu().numpy()))
#                 pred_nii = reconstruct_and_save(subject_id, gad_idx, pred_slices, test_dataset.ras_shape, test_dataset.affine, v, subject_output_dir, f"recon_pred_{v}")
#                 pred_recon = pred_nii.get_fdata()
#                 was_resampled = test_dataset.motion_resampled
#                 pred_recon_original = resample_to_original(pred_recon, test_dataset.ras_shape, was_resampled)
#                 # Use default dtype for saving
#                 input_dtype = np.float32
#                 pred_recon_original = pred_recon_original.astype(input_dtype)
#                 pred_path = os.path.join(subject_output_dir, f'gad_recon_pred_{v}.nii.gz')
#                 nii_out = nib.Nifti1Image(pred_recon_original, test_dataset.affine)
#                 nii_out.header.extensions.clear()
#                 tmp_path = pred_path.replace('.nii.gz', '.nii')
#                 nib.save(nii_out, tmp_path)
#                 with open(tmp_path, 'rb') as f_in, gzip.open(pred_path, 'wb', compresslevel=9) as f_out:
#                     shutil.copyfileobj(f_in, f_out)
#                 os.remove(tmp_path)
#                 pred_volumes.append(pred_recon_original)

#                 # --- Compute metrics for this view in mix mode ---
#                 if test_dataset.has_ground_truth:
#                     free_path, motion_path = _get_nifti_path(data_path, subject_id, gad_idx)
#                     gt_img = tio.ScalarImage(free_path)
#                     motion_img = tio.ScalarImage(motion_path)
#                     gt_vol = (gt_img.data.squeeze(0) - gt_img.data.min()) / (gt_img.data.max() - gt_img.data.min() + 1e-8)
#                     motion_vol = (motion_img.data.squeeze(0) - motion_img.data.min()) / (motion_img.data.max() - motion_img.data.min() + 1e-8)
#                     print(f"[DEBUG] pred_recon_original shape: {pred_recon_original.shape}, gt_vol shape: {gt_vol.shape}")

#                     # If shapes do not match, print a warning and skip metrics for this subject/view
#                     if pred_recon_original.shape != gt_vol.shape:
#                         print(f"[WARNING] Shape mismatch for metrics: pred {pred_recon_original.shape} vs gt {gt_vol.shape}. Skipping metrics.")
#                     else:
#                         pred_recon_tensor = torch.from_numpy(pred_recon_original).unsqueeze(0).unsqueeze(0).float().to(device)
#                         gt_recon_tensor = torch.from_numpy(gt_vol.numpy()).unsqueeze(0).unsqueeze(0).float().to(device)
#                         motion_recon_tensor = torch.from_numpy(motion_vol.numpy()).unsqueeze(0).unsqueeze(0).float().to(device)
#                         pred_vs_free_psnr = psnr(pred_recon_tensor, gt_recon_tensor).item()
#                         pred_vs_free_ssim = ssim_score(pred_recon_tensor, gt_recon_tensor).item()
#                         pred_vs_free_ms_ssim = ms_ssim_score(pred_recon_tensor, gt_recon_tensor).item()
#                         motion_vs_free_psnr = psnr(motion_recon_tensor, gt_recon_tensor).item()
#                         motion_vs_free_ssim = ssim_score(motion_recon_tensor, gt_recon_tensor).item()
#                         motion_vs_free_ms_ssim = ms_ssim_score(motion_recon_tensor, gt_recon_tensor).item()
#                         print(f"    mix-{v} Prediction vs Non-GAD: PSNR={pred_vs_free_psnr:.4f}, SSIM={pred_vs_free_ssim:.4f}, MS-SSIM={pred_vs_free_ms_ssim:.4f}")
#                         print(f"    GAD vs Non-GAD:                PSNR={motion_vs_free_psnr:.4f}, SSIM={motion_vs_free_ssim:.4f}, MS-SSIM={motion_vs_free_ms_ssim:.4f}")
#                         all_metrics.append({
#                             'Subject': subject_id,
#                             'GAD Index': gad_idx,
#                             'View': f'mix-{v}',
#                             'SAP Enabled': enable_sap,
#                             'Number of slices': len(test_dataset),
#                             'Original volume shape': str(test_dataset.ras_shape),
#                             'Prediction vs Non-GAD - PSNR': f"{pred_vs_free_psnr:.4f}",
#                             'Prediction vs Non-GAD - SSIM': f"{pred_vs_free_ssim:.4f}",
#                             'Prediction vs Non-GAD - MS-SSIM': f"{pred_vs_free_ms_ssim:.4f}",
#                             'GAD vs Non-GAD - PSNR': f"{motion_vs_free_psnr:.4f}",
#                             'GAD vs Non-GAD - SSIM': f"{motion_vs_free_ssim:.4f}",
#                             'GAD vs Non-GAD - MS-SSIM': f"{motion_vs_free_ms_ssim:.4f}",
#                             'Improvement - PSNR': f"{pred_vs_free_psnr - motion_vs_free_psnr:.4f}",
#                             'Improvement - SSIM': f"{pred_vs_free_ssim - motion_vs_free_ssim:.4f}",
#                             'Improvement - MS-SSIM': f"{pred_vs_free_ms_ssim - motion_vs_free_ms_ssim:.4f}"
#                         })
#             # Fuse by median if all three views succeeded
#             if all(pv is not None for pv in pred_volumes):
#                 fused_pred = np.median(np.stack(pred_volumes, axis=0), axis=0)
#                 fused_path = os.path.join(subject_output_dir, f'gad_recon_pred_fused.nii.gz')
#                 nii_out_fused = nib.Nifti1Image(fused_pred.astype(input_dtype), test_dataset.affine)
#                 nii_out_fused.header.extensions.clear()
#                 tmp_path_fused = fused_path.replace('.nii.gz', '.nii')
#                 nib.save(nii_out_fused, tmp_path_fused)
#                 with open(tmp_path_fused, 'rb') as f_in, gzip.open(fused_path, 'wb', compresslevel=9) as f_out:
#                     shutil.copyfileobj(f_in, f_out)
#                 os.remove(tmp_path_fused)
#                 print(f"  - Fused prediction saved to {fused_path}")
#                 # --- Compute metrics for fused prediction ---
#                 if test_dataset.has_ground_truth:
#                     free_path, motion_path = _get_nifti_path(data_path, subject_id, gad_idx)
#                     gt_img = tio.ScalarImage(free_path)
#                     motion_img = tio.ScalarImage(motion_path)
#                     gt_vol = (gt_img.data.squeeze(0) - gt_img.data.min()) / (gt_img.data.max() - gt_img.data.min() + 1e-8)
#                     motion_vol = (motion_img.data.squeeze(0) - motion_img.data.min()) / (motion_img.data.max() - motion_img.data.min() + 1e-8)
#                     fused_pred_tensor = torch.from_numpy(fused_pred).unsqueeze(0).unsqueeze(0).float().to(device)
#                     gt_recon_tensor = torch.from_numpy(gt_vol.numpy()).unsqueeze(0).unsqueeze(0).float().to(device)
#                     motion_recon_tensor = torch.from_numpy(motion_vol.numpy()).unsqueeze(0).unsqueeze(0).float().to(device)
#                     pred_vs_free_psnr = psnr(fused_pred_tensor, gt_recon_tensor).item()
#                     pred_vs_free_ssim = ssim_score(fused_pred_tensor, gt_recon_tensor).item()
#                     pred_vs_free_ms_ssim = ms_ssim_score(fused_pred_tensor, gt_recon_tensor).item()
#                     motion_vs_free_psnr = psnr(motion_recon_tensor, gt_recon_tensor).item()
#                     motion_vs_free_ssim = ssim_score(motion_recon_tensor, gt_recon_tensor).item()
#                     motion_vs_free_ms_ssim = ms_ssim_score(motion_recon_tensor, gt_recon_tensor).item()
#                     print(f"    mix-fused Prediction vs Non-GAD: PSNR={pred_vs_free_psnr:.4f}, SSIM={pred_vs_free_ssim:.4f}, MS-SSIM={pred_vs_free_ms_ssim:.4f}")
#                     print(f"    GAD vs Non-GAD:                  PSNR={motion_vs_free_psnr:.4f}, SSIM={motion_vs_free_ssim:.4f}, MS-SSIM={motion_vs_free_ms_ssim:.4f}")
#                     all_metrics.append({
#                         'Subject': subject_id,
#                         'GAD Index': gad_idx,
#                         'View': 'mix-fused',
#                         'SAP Enabled': enable_sap,
#                         'Number of slices': len(test_dataset),
#                         'Original volume shape': str(test_dataset.ras_shape),
#                         'Prediction vs Non-GAD - PSNR': f"{pred_vs_free_psnr:.4f}",
#                         'Prediction vs Non-GAD - SSIM': f"{pred_vs_free_ssim:.4f}",
#                         'Prediction vs Non-GAD - MS-SSIM': f"{pred_vs_free_ms_ssim:.4f}",
#                         'GAD vs Non-GAD - PSNR': f"{motion_vs_free_psnr:.4f}",
#                         'GAD vs Non-GAD - SSIM': f"{motion_vs_free_ssim:.4f}",
#                         'GAD vs Non-GAD - MS-SSIM': f"{motion_vs_free_ms_ssim:.4f}",
#                         'Improvement - PSNR': f"{pred_vs_free_psnr - motion_vs_free_psnr:.4f}",
#                         'Improvement - SSIM': f"{pred_vs_free_ssim - motion_vs_free_ssim:.4f}",
#                         'Improvement - MS-SSIM': f"{pred_vs_free_ms_ssim - motion_vs_free_ms_ssim:.4f}"
#                     })
#                 continue  # Skip the rest of the loop for mix mode
            
#         try:
#             test_dataset = NiftiTestDataset(subject_id, gad_idx, view, data_path, enable_sap)
#             test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#         except FileNotFoundError as e:
#             print(f"Skipping {subject_id}: {e}")
#             continue
        
#         print(f"\n--- Processing Subject: {subject_id}, Slices: {len(test_dataset)} ---")

#         pred_slices, gt_slices, gad_slices = [], [], []
        
#         for i, (x, y) in enumerate(test_loader):
#             if enable_sap:
#                 # DataLoader already provides correct format, just move to device
#                 x_tensors = [t.to(device) for t in x]
#                 with torch.no_grad():
#                     pred = model.model(*x_tensors)
#                 gad_slices.append((i, x[1].squeeze().cpu().numpy()))  # x[1] is the current slice
#             else:
#                 # DataLoader already provides correct format, just move to device
#                 x_tensor = x.to(device)
#                 with torch.no_grad():
#                     pred = model.model(x_tensor)
#                 gad_slices.append((i, x.squeeze().cpu().numpy()))  # x is the current slice
            
#             pred_slices.append((i, pred.squeeze().cpu().numpy()))
#             gt_slices.append((i, y.squeeze().cpu().numpy()))
        
#         # --- Plotting with Metrics ---
#         if args.generate_plots:
#             central_idx = len(test_dataset) // 2
#             gad_slice_np = gad_slices[central_idx][1]
#             pred_slice_np = pred_slices[central_idx][1]
#             gt_slice_np = gt_slices[central_idx][1]

#             gad_tensor = torch.from_numpy(gad_slice_np).unsqueeze(0).unsqueeze(0).float().to(device)
#             pred_tensor = torch.from_numpy(pred_slice_np).unsqueeze(0).unsqueeze(0).float().to(device)
#             gt_tensor = torch.from_numpy(gt_slice_np).unsqueeze(0).unsqueeze(0).float().to(device)

#             psnr_gad = psnr(gad_tensor, gt_tensor).item()
#             ssim_gad = ssim_score(gad_tensor, gt_tensor).item()
#             ms_ssim_gad = ms_ssim_score(gad_tensor, gt_tensor).item()
#             psnr_pred = psnr(pred_tensor, gt_tensor).item()
#             ssim_pred = ssim_score(pred_tensor, gt_tensor).item()
#             ms_ssim_pred = ms_ssim_score(pred_tensor, gt_tensor).item()
            
#             images = {
#                 f"GAD Input\nPSNR: {psnr_gad:.2f} | SSIM: {ssim_gad:.4f}\nMS-SSIM: {ms_ssim_gad:.4f}": gad_slice_np,
#                 f"Prediction\nPSNR: {psnr_pred:.2f} | SSIM: {ssim_pred:.4f}\nMS-SSIM: {ms_ssim_pred:.4f}": pred_slice_np,
#                 "Ground Truth (Non-GAD)": gt_slice_np
#             }
#             fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#             fig.suptitle(f'Subject: {subject_id}, Slice: {central_idx} ({view})', fontsize=16)

#             for ax, (title, img_np) in zip(axes, images.items()):
#                 ax.imshow(img_np.T, cmap='gray', origin='lower')
#                 ax.axis('off'); ax.set_title(title)

#             plot_path = os.path.join(subject_output_dir, f'gad_plot.png')
#             plt.savefig(plot_path, bbox_inches='tight'); plt.close()
#             print(f"  - Saved comparison plot to {plot_path}")

#         # --- Reconstruction and Volume Metrics ---
#         print("  - Reconstructing volumes...")
#         pred_nii = reconstruct_and_save(subject_id, gad_idx, pred_slices, test_dataset.ras_shape, test_dataset.affine, view, subject_output_dir, "recon_pred")
#         pred_recon = pred_nii.get_fdata()
#         was_resampled = test_dataset.motion_resampled
#         pred_recon_original = resample_to_original(pred_recon, test_dataset.ras_shape, was_resampled)

#         # Save the prediction in original dimensions (single pass)
#         input_dtype = np.float32
#         pred_recon_original = pred_recon_original.astype(input_dtype)
#         pred_original_path = os.path.join(subject_output_dir, f'gad_recon_pred.nii.gz')
#         nii_out = nib.Nifti1Image(pred_recon_original, test_dataset.affine)
#         nii_out.header.extensions.clear()
#         tmp_path = pred_original_path.replace('.nii.gz', '.nii')
#         nib.save(nii_out, tmp_path)
#         with open(tmp_path, 'rb') as f_in, gzip.open(pred_original_path, 'wb', compresslevel=9) as f_out:
#             shutil.copyfileobj(f_in, f_out)
#         os.remove(tmp_path)

#         # --- Compute metrics between prediction and ground truth (non-GAD) ---
#         # Motivation: The following logic is directly inspired by the training dataset (see NiftiSliceDataset in dataset.py).
#         # Both the predicted and ground truth (non-GAD) volumes are loaded, normalized, and padded/cropped in the exact same way as during training.
#         # This ensures that the computed metrics (PSNR, SSIM, MS-SSIM) are directly comparable to those used during training and validation.

#         if test_dataset.has_ground_truth:
#             free_path, motion_path = _get_nifti_path(data_path, subject_id, gad_idx)
#             gt_img = tio.ScalarImage(free_path)
#             motion_img = tio.ScalarImage(motion_path)
#             # Normalize volumes as in training
#             gt_vol = (gt_img.data.squeeze(0) - gt_img.data.min()) / (gt_img.data.max() - gt_img.data.min() + 1e-8)
#             motion_vol = (motion_img.data.squeeze(0) - motion_img.data.min()) / (motion_img.data.max() - motion_img.data.min() + 1e-8)
#             print(f"[DEBUG] pred_recon_original shape: {pred_recon_original.shape}, gt_vol shape: {gt_vol.shape}")

#             # If shapes do not match, print a warning and skip metrics for this subject/view
#             if pred_recon_original.shape != gt_vol.shape:
#                 print(f"[WARNING] Shape mismatch for metrics: pred {pred_recon_original.shape} vs gt {gt_vol.shape}. Skipping metrics.")
#             else:
#                 pred_recon_tensor = torch.from_numpy(pred_recon_original).unsqueeze(0).unsqueeze(0).float().to(device)
#                 gt_recon_tensor = torch.from_numpy(gt_vol.numpy()).unsqueeze(0).unsqueeze(0).float().to(device)
#                 motion_recon_tensor = torch.from_numpy(motion_vol.numpy()).unsqueeze(0).unsqueeze(0).float().to(device)
#                 # Calculate metrics: prediction vs non-GAD and GAD vs non-GAD
#                 pred_vs_free_psnr = psnr(pred_recon_tensor, gt_recon_tensor).item()
#                 pred_vs_free_ssim = ssim_score(pred_recon_tensor, gt_recon_tensor).item()
#                 pred_vs_free_ms_ssim = ms_ssim_score(pred_recon_tensor, gt_recon_tensor).item()
#                 motion_vs_free_psnr = psnr(motion_recon_tensor, gt_recon_tensor).item()
#                 motion_vs_free_ssim = ssim_score(motion_recon_tensor, gt_recon_tensor).item()
#                 motion_vs_free_ms_ssim = ms_ssim_score(motion_recon_tensor, gt_recon_tensor).item()
#                 # Print metrics for this subject
#                 print(f"    Prediction vs Non-GAD: PSNR={pred_vs_free_psnr:.4f}, SSIM={pred_vs_free_ssim:.4f}, MS-SSIM={pred_vs_free_ms_ssim:.4f}")
#                 print(f"    GAD vs Non-GAD:        PSNR={motion_vs_free_psnr:.4f}, SSIM={motion_vs_free_ssim:.4f}, MS-SSIM={motion_vs_free_ms_ssim:.4f}")
#                 # Collect metrics for CSV
#                 all_metrics.append({
#                     'Subject': subject_id,
#                     'GAD Index': gad_idx,
#                     'View': view,
#                     'SAP Enabled': enable_sap,
#                     'Number of slices': len(test_dataset),
#                     'Original volume shape': str(test_dataset.ras_shape),
#                     'Prediction vs Non-GAD - PSNR': f"{pred_vs_free_psnr:.4f}",
#                     'Prediction vs Non-GAD - SSIM': f"{pred_vs_free_ssim:.4f}",
#                     'Prediction vs Non-GAD - MS-SSIM': f"{pred_vs_free_ms_ssim:.4f}",
#                     'GAD vs Non-GAD - PSNR': f"{motion_vs_free_psnr:.4f}",
#                     'GAD vs Non-GAD - SSIM': f"{motion_vs_free_ssim:.4f}",
#                     'GAD vs Non-GAD - MS-SSIM': f"{motion_vs_free_ms_ssim:.4f}",
#                     'Improvement - PSNR': f"{pred_vs_free_psnr - motion_vs_free_psnr:.4f}",
#                     'Improvement - SSIM': f"{pred_vs_free_ssim - motion_vs_free_ssim:.4f}",
#                     'Improvement - MS-SSIM': f"{pred_vs_free_ms_ssim - motion_vs_free_ms_ssim:.4f}"
#                 })
#                 print(f"  - Processed {subject_id} with metrics")
#         else:
#             print(f"  - Processed {subject_id} (no ground truth available for metrics)")

#     # Save metrics to CSV in the main output directory
#     csv_path = os.path.join(args.output_dir, 'all_metrics.csv')
#     with open(csv_path, 'w', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=['Subject', 'GAD Index', 'View', 'SAP Enabled', 'Number of slices', 'Original volume shape', 'Prediction vs Non-GAD - PSNR', 'Prediction vs Non-GAD - SSIM', 'Prediction vs Non-GAD - MS-SSIM', 'GAD vs Non-GAD - PSNR', 'GAD vs Non-GAD - SSIM', 'GAD vs Non-GAD - MS-SSIM', 'Improvement - PSNR', 'Improvement - SSIM', 'Improvement - MS-SSIM'])
#         writer.writeheader()
#         writer.writerows(all_metrics)
    
#     # --- Compute and save averages for requested views ---
#     def safe_float(x):
#         try:
#             return float(x)
#         except:
#             return np.nan

#     # Views to average
#     views_to_average = [
#         ('GAD vs Non-GAD', 'GAD vs Non-GAD'),
#         ('mix-fused', 'mix-fused'),
#         ('mix-sagittal', 'mix-sagittal'),
#         ('mix-coronal', 'mix-coronal'),
#         ('mix-axial', 'mix-axial'),
#     ]

#     # Prepare rows for averages
#     average_rows = []
#     for label, view_key in views_to_average:
#         # For 'GAD vs Non-GAD', average across all rows (regardless of view)
#         if label == 'GAD vs Non-GAD':
#             relevant = all_metrics
#         else:
#             relevant = [row for row in all_metrics if row['View'] == view_key]
#         if not relevant:
#             continue
#         avg_pred_psnr = np.nanmean([safe_float(row['Prediction vs Non-GAD - PSNR']) for row in relevant])
#         avg_pred_ssim = np.nanmean([safe_float(row['Prediction vs Non-GAD - SSIM']) for row in relevant])
#         avg_pred_ms_ssim = np.nanmean([safe_float(row['Prediction vs Non-GAD - MS-SSIM']) for row in relevant])
#         avg_gad_psnr = np.nanmean([safe_float(row['GAD vs Non-GAD - PSNR']) for row in relevant])
#         avg_gad_ssim = np.nanmean([safe_float(row['GAD vs Non-GAD - SSIM']) for row in relevant])
#         avg_gad_ms_ssim = np.nanmean([safe_float(row['GAD vs Non-GAD - MS-SSIM']) for row in relevant])
#         avg_impr_psnr = np.nanmean([safe_float(row['Improvement - PSNR']) for row in relevant])
#         avg_impr_ssim = np.nanmean([safe_float(row['Improvement - SSIM']) for row in relevant])
#         avg_impr_ms_ssim = np.nanmean([safe_float(row['Improvement - MS-SSIM']) for row in relevant])
#         average_rows.append({
#             'Subject': 'Average',
#             'GAD Index': '',
#             'View': view_key,
#             'SAP Enabled': '',
#             'Number of slices': '',
#             'Original volume shape': '',
#             'Prediction vs Non-GAD - PSNR': f"{avg_pred_psnr:.4f}",
#             'Prediction vs Non-GAD - SSIM': f"{avg_pred_ssim:.4f}",
#             'Prediction vs Non-GAD - MS-SSIM': f"{avg_pred_ms_ssim:.4f}",
#             'GAD vs Non-GAD - PSNR': f"{avg_gad_psnr:.4f}",
#             'GAD vs Non-GAD - SSIM': f"{avg_gad_ssim:.4f}",
#             'GAD vs Non-GAD - MS-SSIM': f"{avg_gad_ms_ssim:.4f}",
#             'Improvement - PSNR': f"{avg_impr_psnr:.4f}",
#             'Improvement - SSIM': f"{avg_impr_ssim:.4f}",
#             'Improvement - MS-SSIM': f"{avg_impr_ms_ssim:.4f}"
#         })

#     # Append averages to CSV
#     with open(csv_path, 'a', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=['Subject', 'GAD Index', 'View', 'SAP Enabled', 'Number of slices', 'Original volume shape', 'Prediction vs Non-GAD - PSNR', 'Prediction vs Non-GAD - SSIM', 'Prediction vs Non-GAD - MS-SSIM', 'GAD vs Non-GAD - PSNR', 'GAD vs Non-GAD - SSIM', 'GAD vs Non-GAD - MS-SSIM', 'Improvement - PSNR', 'Improvement - SSIM', 'Improvement - MS-SSIM'])
#         for row in average_rows:
#             writer.writerow(row)

#     print(f"  - Metrics saved to {csv_path} (including averages)") 

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