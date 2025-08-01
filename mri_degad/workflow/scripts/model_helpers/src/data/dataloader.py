import os
from src.data.dataset import NiftiSliceDataset
# from dataset import NiftiSliceDataset

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    data_path = "/cifs/khan_new/trainees/msnyde26/degad/work/"
    output_dir = "dataloader_test_plots_v2"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving dataloader test plots to: {output_dir}")

    if not os.path.exists(data_path):
        print(f"Data directory '{data_path}' not found.")
        exit()

    all_subjects = [d for d in os.listdir(data_path) if d.startswith('sub') and os.path.isdir(os.path.join(data_path, d))]
    if not all_subjects:
        print("No subjects found in data directory.")
        exit()

    for subject in all_subjects:
        print(f"Processing subject: {subject}")
        for view in ["Sagittal", "Axial", "Coronal"]:
            print(f"  Testing view: {view}")
            try:
                dataset = NiftiSliceDataset(
                    data_path,
                    [subject],
                    view=view,
                    data_id='Motion',
                    enable_SAP=True
                )

                if len(dataset) == 0:
                    print(f"    No slices found for view {view}. Skipping.")
                    continue

                slice_idx = len(dataset) // 3
                x, y = dataset[slice_idx]
                slice_tuple = dataset.slice_tuples[slice_idx]

                motion_before, motion_current, motion_after = x
                
                images = {
                    "Motion Before": motion_before,
                    "Motion Current": motion_current,
                    "Motion After": motion_after,
                    "Ground Truth": y
                }

                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                fig.suptitle(f'Subject: {subject}, View: {view}, Slice: {slice_tuple[2]}', fontsize=16)

                for ax, (title, img_tensor) in zip(axes, images.items()):
                    img_np = img_tensor.cpu().numpy().squeeze()
                    ax.imshow(img_np, cmap='gray', origin='lower')
                    ax.set_title(title)
                    ax.axis('off')

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plot_path = os.path.join(output_dir, f'{subject}_{view}_slice.png')
                plt.savefig(plot_path)
                plt.close(fig)
                print(f"    Saved plot to {plot_path}")
            except Exception as e:
                print(f"    Could not process view {view} for subject {subject}. Error: {e}") 
        break
