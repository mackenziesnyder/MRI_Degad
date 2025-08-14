import os
import torch
import torchio as tio
from torch.utils.data import Dataset
import numpy as np

def resample_to_original(volume, original_shape, was_resampled=False, view=None):
    """Resamples or unpads a volume back to original dimensions to ensure output corresponds to input."""
    if was_resampled:
        # Resample back to original shape
        print(f"view: {view}")
        print(f"    Resampling from {volume.shape} to {original_shape}")
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        volume_resampled = torch.nn.functional.interpolate(
            volume_tensor,
            size=original_shape,
            mode='trilinear',
            align_corners=True,
            # antialias=True
        ).squeeze(0).squeeze(0).numpy()  # Remove batch and channel dims
        print(f"    Resampled shape: {volume_resampled.shape}")
        return volume_resampled
    else:
        # Unpad to original shape - reverse the padding/cropping from training
        print(f"    Unpadding from {volume.shape} to {original_shape}")
        
        # The training dataset does:
        # 1. Pad to (261, 263, 256) 
        # 2. Center crop to (256, 256, 256)
        # We need to reverse this process
        
        # Step 1: Reverse the center cropping
        # The center crop takes the middle 256x256x256 from the padded volume
        target_shape = (261, 263, 256)  # Original padded shape
        crop_shape = (256, 256, 256)    # What was cropped to
        
        # Calculate the crop start positions that were used during training
        start = [(target_shape[i] - crop_shape[i]) // 2 for i in range(3)]
        end = [start[i] + crop_shape[i] for i in range(3)]
        
        # Create a larger volume filled with zeros (the padded size)
        unpadded_volume = np.zeros(target_shape, dtype=volume.dtype)
        
        for i in range(3):
            end[i] = start[i] + volume.shape[i]
        # Place the cropped volume back in the center
        unpadded_volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = volume
        
        # Step 2: Remove the padding to get back to original dimensions
        # Calculate how much padding was added during training
        pad_needed = [(0, max(0, target_shape[i] - original_shape[i])) for i in range(3)]
        
        # Remove the padding from each dimension
        final_volume = unpadded_volume
        for i in range(3):
            if pad_needed[i][1] > 0:  # If padding was added
                if i == 0:  # Width dimension
                    final_volume = final_volume[:original_shape[0], :, :]
                elif i == 1:  # Height dimension
                    final_volume = final_volume[:, :original_shape[1], :]
                elif i == 2:  # Depth dimension
                    final_volume = final_volume[:, :, :original_shape[2]]
        
        print(f"    Unpadded shape: {final_volume.shape}")

    if view == 'axial':
        # Your dataset permuted axial slices with (2,0,1), so invert here
        volume = np.transpose(volume, (1, 2, 0))  # from (slice, W, H) to (W, H, slice)
    elif view == 'coronal':
        # Dataset permuted coronal slices with (1,0,2)
        volume = np.transpose(volume, (1, 0, 2))
    elif view == 'sagittal':
        # Sagittal may not be permuted, but check your dataset logic
        # If no permutation used, no change needed
        pass
    else:
        raise ValueError(f"Unknown view: {view}")
    
    return final_volume

def _pad_and_crop_volume(vol):
    """Pad to (261, 263, 256) then center crop to (256, 256, 256) - matches training dataset."""
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

# def _get_nifti_path(data_path, subject, number_of_motion):
#     """Get NIfTI paths using the same structure as training dataset for DEGAD."""
#     base_path = os.path.join(data_path, subject, 'ses-pre', 'normalize')
    
#     # For DEGAD: we have GAD (input) and Non-GAD (ground truth)
#     # The training dataset uses GAD as input and Non-GAD as target
#     gad_path = os.path.join(base_path, f'{subject}_ses-pre_acq-gad_run-01_desc-normalize_minmax_T1w.nii.gz')
#     nongad_path = os.path.join(base_path, f'{subject}_ses-pre_acq-nongad_run-01_desc-normalized_zscore_T1w.nii.gz')
    
#     # For inference, we'll use GAD as the input volume and Non-GAD as ground truth
#     if os.path.exists(gad_path) and os.path.exists(nongad_path):
#         return nongad_path, gad_path  # Return (ground_truth, input) for consistency
#     else:
#         return None, None

class NiftiTestDataset(Dataset):
    """
    A PyTorch Dataset for testing that loads a single subject's GAD volume,
    normalizes and processes it using the EXACT same logic as the training dataset.
    Supports 'mix' mode for inference: if view='mix', the dataset will contain slices from all three views (axial, sagittal, coronal).
    """
    def __init__(self,  view, image_path, enable_sap=True):
        valid_views = ['axial', 'sagittal', 'coronal', 'mix']
        if view not in valid_views:
            raise ValueError(f"Invalid view '{view}' for NiftiTestDataset. Must be one of: {valid_views}.")
        self.view = view.lower()
        self.enable_sap = enable_sap

        # Load, normalize, and process the volumes using EXACT same logic as training dataset
        # free_path, motion_path = _get_nifti_path(data_path, subject_id, motion_idx)
        # print(f"[DEBUG] Subject: {subject_id}")
        # print(f"[DEBUG] Checking GAD path: {motion_path}, exists: {os.path.exists(motion_path) if motion_path else 'N/A'}")
        # print(f"[DEBUG] Checking Non-GAD (ground truth) path: {free_path}, exists: {os.path.exists(free_path) if free_path else 'N/A'}")
        # if not motion_path:
        #     raise FileNotFoundError(f"Could not find GAD NIfTI file for subject {subject_id}")

        # Load GAD volume (input) - required - EXACT same as training
        motion_img = tio.ScalarImage(image_path)
        self.affine = motion_img.affine
        self.ras_shape = motion_img.shape[1:]  # Original (W, H, D)
        motion_vol = motion_img.data.squeeze(0)  # EXACT same as training
        motion_vol = (motion_vol - motion_vol.min()) / (motion_vol.max() - motion_vol.min() + 1e-8)  # EXACT same normalization
        self.motion_padded = _pad_and_crop_volume(motion_vol)  # EXACT same padding/cropping
        self.motion_resampled = False  # We use padding/cropping, not resampling

        # Load Non-GAD volume (ground truth) - optional for testing - EXACT same as training
        # if free_path and os.path.exists(free_path):
        #     # print(f"[DEBUG] Ground truth found for subject {subject_id} at {free_path}")
        #     free_img = tio.ScalarImage(free_path)
        #     free_vol = free_img.data.squeeze(0)  # EXACT same as training
        #     free_vol = (free_vol - free_vol.min()) / (free_vol.max() - free_vol.min() + 1e-8)  # EXACT same normalization
        #     self.free_padded = _pad_and_crop_volume(free_vol)  # EXACT same padding/cropping
        #     self.free_resampled = False
        #     self.has_ground_truth = True
        # else:
        # print(f"[DEBUG] Ground truth NOT found for subject {subject_id}. Path checked: {free_path}")
        self.free_padded = torch.zeros_like(self.motion_padded)
        self.free_resampled = False
        self.has_ground_truth = False

        # Build slice_tuples for mix or single view - EXACT same as training
        self.slice_tuples = self._build_slice_tuples()

    def _build_slice_tuples(self):
        slice_tuples = []
        W, H, D = self.motion_padded.shape
        if self.view == 'mix':
            # EXACT same slice range as training dataset
            for slice_id in range(1, D - 2):
                slice_tuples.append(('axial', slice_id))
            for slice_id in range(1, W - 2):
                slice_tuples.append(('sagittal', slice_id))
            for slice_id in range(1, H - 2):
                slice_tuples.append(('coronal', slice_id))
        else:
            if self.view == 'sagittal':
                num_slices = W
            elif self.view == 'coronal':
                num_slices = H
            elif self.view == 'axial':
                num_slices = D
            else:
                raise ValueError(f"Unknown view: {self.view}")
            # EXACT same slice range as training dataset
            for slice_id in range(1, num_slices - 2):
                slice_tuples.append((self.view, slice_id))
        return slice_tuples

    def __len__(self):
        return len(self.slice_tuples)

    def __getitem__(self, idx):
        view_name, slice_id = self.slice_tuples[idx]
        
        # Use the EXACT same slicing logic as training dataset
        if view_name == 'sagittal':
            # EXACT same indexing as training dataset
            free_slice = self.free_padded[slice_id+1:slice_id+2, :, :]
            motion_slice = self.motion_padded[slice_id+1:slice_id+2, :, :]
            motion_before_slice = self.motion_padded[slice_id:slice_id+1, :, :]
            motion_after_slice = self.motion_padded[slice_id+2:slice_id+3, :, :]
        elif view_name == 'coronal':
            # EXACT same indexing as training dataset
            free_slice = self.free_padded[:, slice_id+1:slice_id+2, :]
            motion_slice = self.motion_padded[:, slice_id+1:slice_id+2, :]
            motion_before_slice = self.motion_padded[:, slice_id:slice_id+1, :]
            motion_after_slice = self.motion_padded[:, slice_id+2:slice_id+3, :]
            # EXACT same permutation as training dataset
            free_slice = free_slice.permute(1, 0, 2)
            motion_slice = motion_slice.permute(1, 0, 2)
            motion_before_slice = motion_before_slice.permute(1, 0, 2)
            motion_after_slice = motion_after_slice.permute(1, 0, 2)
        elif view_name == 'axial':
            # EXACT same indexing as training dataset
            free_slice = self.free_padded[:, :, slice_id+1:slice_id+2]
            motion_slice = self.motion_padded[:, :, slice_id+1:slice_id+2]
            motion_before_slice = self.motion_padded[:, :, slice_id:slice_id+1]
            motion_after_slice = self.motion_padded[:, :, slice_id+2:slice_id+3]
            # EXACT same permutation as training dataset
            free_slice = free_slice.permute(2, 0, 1)
            motion_slice = motion_slice.permute(2, 0, 1)
            motion_before_slice = motion_before_slice.permute(2, 0, 1)
            motion_after_slice = motion_after_slice.permute(2, 0, 1)
        else:
            raise ValueError(f"Unknown view: {view_name}")
        
        # EXACT same type conversion as training dataset
        free_slice = free_slice.float()
        motion_slice = motion_slice.float()
        motion_before_slice = motion_before_slice.float()
        motion_after_slice = motion_after_slice.float()
        
        # EXACT same return format as training dataset
        if self.enable_sap:
            # Return SAP format: (before, current, after), ground_truth
            return (motion_before_slice, motion_slice, motion_after_slice), free_slice
        else:
            # Return single slice format: input, ground_truth
            return motion_slice, free_slice 