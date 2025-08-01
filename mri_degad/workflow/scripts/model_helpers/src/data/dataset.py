import os
import torch
from torch.utils.data import Dataset
import torchio as tio
import numpy as np
import random
import threading

class NiftiSliceDataset(Dataset):
    """
    Optimized PyTorch Dataset for loading slices from NIfTI files with selective in-memory caching, using TorchIO for image loading.
    TorchIO loads images as (C, D, H, W) and reorients to RAS+ by default.
    This class squeezes the channel dimension to work with (D, H, W) for slicing.
    All slicing assumes (W, H, D) axis order (RAS+ orientation: Left-Right, Inferior-Superior, Posterior-Anterior).
    
    Args:
        data_path (str): Path to the dataset directory
        subjects (list): List of subject IDs
        view (str): View direction ('axial', 'sagittal', 'coronal')
        data_id (str): Data identifier ('motion', etc.)
        crop (bool): Whether to crop volumes
        enable_SAP (bool): Whether to enable SAP (Slice Adjacent Processing)
        preload_probability (float): Probability of preloading each volume (0.0 to 1.0). 
                                   Default 1.0 loads all volumes, 0.5 loads ~50% of volumes.
    """
    def __init__(self, data_path, subjects, view='Axial', data_id='Motion', crop=False, enable_SAP=True, preload_probability=1.0):
        self.data_path = data_path
        self.subjects = subjects
        self.view = view.lower()
        self.data_id = data_id.lower()
        self.crop = crop
        self.enable_SAP = enable_SAP
        self.preload_probability = preload_probability
        self.volume_cache = {}
        self._cache_lock = threading.Lock()  # Thread safety for cache access
        
        # Preload volumes selectively based on probability
        self._preload_volumes()
        self.slice_tuples = self._build_slice_tuples()
        
        # Log preloading statistics
        total_possible_volumes = len(subjects) * 2  # 2 motion volumes per subject
        preloaded_volumes = len(self.volume_cache)
        print(f"NiftiSliceDataset: Preloaded {preloaded_volumes}/{total_possible_volumes} volumes "
              f"(probability={preload_probability:.2f}) from {len(self.subjects)} subjects. "
              f"All available volumes will be accessible during training. "
              f"Generated {len(self.slice_tuples)} slices.")

    def _get_nifti_path(self, subject):
        base_path = os.path.join(self.data_path, subject, 'ses-pre', 'normalize')
        gad_path = os.path.join(base_path, f'{subject}_ses-pre_acq-gad_run-01_desc-normalize_minmax_T1w.nii.gz')
        nongad_path = os.path.join(base_path, f'{subject}_ses-pre_acq-nongad_run-01_desc-normalized_zscore_T1w.nii.gz')
        if os.path.exists(gad_path) and os.path.exists(nongad_path):
            return nongad_path, gad_path
        else:
            return None, None

    def _pad_and_crop_volume(self, vol):
        # Pad to (261, 263, 256)
        target_shape = (261, 263, 256)
        pad_needed = [(0, max(0, target_shape[i] - vol.shape[i])) for i in range(3)]
        vol = torch.nn.functional.pad(vol, (0, pad_needed[2][1], 0, pad_needed[1][1], 0, pad_needed[0][1]))
        # Center crop to (256, 256, 256)
        crop_shape = (256, 256, 256)
        start = [(vol.shape[i] - crop_shape[i]) // 2 for i in range(3)]
        end = [start[i] + crop_shape[i] for i in range(3)]
        vol = vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        return vol

    # def _apply_rotation90(self, vol):
    #     # Rotate 90 degrees (k=1) in-plane (axes 1,2)
    #     vol = torch.rot90(vol, 3, [1, 2])
    #     return vol

    def _preload_volumes(self):
        """Selectively preload volumes based on preload_probability."""
        for subject in self.subjects:
            if random.random() > self.preload_probability:
                continue
            key = (subject, 1)
            free_path, gad_path = self._get_nifti_path(subject)
            if free_path and gad_path:
                try:
                    free_img = tio.ScalarImage(free_path)
                    gad_img = tio.ScalarImage(gad_path)
                    free = free_img.data.squeeze(0)
                    gad = gad_img.data.squeeze(0)
                    if free.ndim != 3 or gad.ndim != 3:
                        print(f"[WARNING] Loaded tensor is not 3D after squeeze: free shape={free.shape}, gad shape={gad.shape}")
                    free = (free - free.min()) / (free.max() - free.min() + 1e-8)
                    gad = (gad - gad.min()) / (gad.max() - gad.min() + 1e-8)
                    # free = self._apply_rotation90(free)
                    # gad = self._apply_rotation90(gad)
                    free = self._pad_and_crop_volume(free)
                    gad = self._pad_and_crop_volume(gad)
                    with self._cache_lock:
                        self.volume_cache[key] = (free, gad)
                except Exception as e:
                    print(f"[WARNING] Failed to load volume for {subject}: {e}")

    def _load_volume_on_demand(self, subject, _):
        key = (subject, 1)
        with self._cache_lock:
            if key in self.volume_cache:
                return self.volume_cache[key]
        free_path, gad_path = self._get_nifti_path(subject)
        if free_path and gad_path:
            try:
                free_img = tio.ScalarImage(free_path)
                gad_img = tio.ScalarImage(gad_path)
                free = free_img.data.squeeze(0)
                gad = gad_img.data.squeeze(0)
                free = (free - free.min()) / (free.max() - free.min() + 1e-8)
                gad = (gad - gad.min()) / (gad.max() - gad.min() + 1e-8)
                # free = self._apply_rotation90(free)
                # gad = self._apply_rotation90(gad)
                free = self._pad_and_crop_volume(free)
                gad = self._pad_and_crop_volume(gad)
                with self._cache_lock:
                    self.volume_cache[key] = (free, gad)
                return free, gad
            except Exception as e:
                print(f"[ERROR] Failed to load volume on-demand for {subject}: {e}")
                dummy_tensor = torch.zeros((256, 256, 256), dtype=torch.float32)
                return dummy_tensor, dummy_tensor
        else:
            dummy_tensor = torch.zeros((256, 256, 256), dtype=torch.float32)
            return dummy_tensor, dummy_tensor

    def _build_slice_tuples(self):
        slice_tuples = []
        for subject in self.subjects:
            free_path, gad_path = self._get_nifti_path(subject)
            if not (free_path and gad_path):
                print(f"Warning: No valid gad/free volumes found for subject {subject}. Skipping.")
                continue
            try:
                if (subject, 1) in self.volume_cache:
                    free_vol, _ = self.volume_cache[(subject, 1)]
                else:
                    free_img = tio.ScalarImage(free_path)
                    free_vol = free_img.data.squeeze(0)
                    free_vol = (free_vol - free_vol.min()) / (free_vol.max() - free_vol.min() + 1e-8)
                    free_vol = self._pad_and_crop_volume(free_vol)
                W, H, D = free_vol.shape
                if self.view == 'mix':
                    for slice_id in range(1, D - 2):
                        slice_tuples.append((subject, 1, slice_id, 'axial'))
                    for slice_id in range(1, W - 2):
                        slice_tuples.append((subject, 1, slice_id, 'sagittal'))
                    for slice_id in range(1, H - 2):
                        slice_tuples.append((subject, 1, slice_id, 'coronal'))
                else:
                    num_slices = 0
                    if self.view == 'sagittal':
                        num_slices = W
                    elif self.view == 'coronal':
                        num_slices = H
                    elif self.view == 'axial':
                        num_slices = D
                    for slice_id in range(1, num_slices - 2):
                        slice_tuples.append((subject, 1, slice_id))
            except Exception as e:
                print(f"Warning: Failed to process subject {subject}: {e}. Skipping.")
                continue
        return slice_tuples

    def __len__(self):
        return len(self.slice_tuples)

    def __getitem__(self, idx):
        if self.view == 'mix':
            subject, i, slice_id, view_name = self.slice_tuples[idx]
        else:
            subject, i, slice_id = self.slice_tuples[idx]
            view_name = self.view
        try:
            free, gad = self._load_volume_on_demand(subject, i)
        except Exception as e:
            raise RuntimeError(f"Failed to load volume for {subject} motion {i}: {e}")
        if torch.all(free == 0) and torch.all(gad == 0):
            raise RuntimeError(f"Loaded volume for {subject} motion {i} is all zeros (likely failed to load). Aborting.")
        try:
            if view_name == 'sagittal':
                free_slice = free[slice_id+1:slice_id+2, :, :]
                gad_slice = gad[slice_id+1:slice_id+2, :, :]
                gad_before_slice = gad[slice_id:slice_id+1, :, :]
                gad_after_slice = gad[slice_id+2:slice_id+3, :, :]
            elif view_name == 'coronal':
                free_slice = free[:, slice_id+1:slice_id+2, :]
                gad_slice = gad[:, slice_id+1:slice_id+2, :]
                gad_before_slice = gad[:, slice_id:slice_id+1, :]
                gad_after_slice = gad[:, slice_id+2:slice_id+3, :]
                free_slice = free_slice.permute(1, 0, 2)
                gad_slice = gad_slice.permute(1, 0, 2)
                gad_before_slice = gad_before_slice.permute(1, 0, 2)
                gad_after_slice = gad_after_slice.permute(1, 0, 2)
            elif view_name == 'axial':
                free_slice = free[:, :, slice_id+1:slice_id+2]
                gad_slice = gad[:, :, slice_id+1:slice_id+2]
                gad_before_slice = gad[:, :, slice_id:slice_id+1]
                gad_after_slice = gad[:, :, slice_id+2:slice_id+3]
                free_slice = free_slice.permute(2, 0, 1)
                gad_slice = gad_slice.permute(2, 0, 1)
                gad_before_slice = gad_before_slice.permute(2, 0, 1)
                gad_after_slice = gad_after_slice.permute(2, 0, 1)
            else:
                raise ValueError(f"Unknown view: {view_name}")
            free_slice = free_slice.float()
            gad_slice = gad_slice.float()
            gad_before_slice = gad_before_slice.float()
            gad_after_slice = gad_after_slice.float()
            if self.enable_SAP:
                return (gad_before_slice, gad_slice, gad_after_slice), free_slice
            else:
                return gad_slice, free_slice
        except Exception as e:
            raise RuntimeError(f"Failed to process slice {slice_id} for {subject} motion {i} (view {view_name}): {e}") 