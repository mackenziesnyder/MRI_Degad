# script that will apply the model 

import monai 
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityd)
import nibabel as nib
import numpy as np
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader
import torch
import os
import glob
import math

def get_model_file(model_dir):
    model_files = glob.glob(os.path.join(model_dir, "*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model file found in {model_dir}!")
    model_file = model_files[0]
    return model_file

def train_model(model_file, t1w_gad, t1w_degad):

    args = model_file.split('_')[4:7]
    filter = int(args[0].split('-')[1])
    cnn_depth= int(args[1].split('-')[1])
    layer_per_block = int(args[2].split('-')[1])
    bottleneck = cnn_depth

    channels = ()
    for i in range(cnn_depth):
        # needed to divide the filter by 2 here
        channels += tuple(filter // 2 for _ in range(layer_per_block))
        filter *= 2 
    channels += tuple(filter // 2 for _ in range(bottleneck))

    strides = ()
    for i in range(cnn_depth):
        strides += (2,) + (1,)*(layer_per_block -1)
    strides += (bottleneck-1) * (1,)

    CNN = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=channels, 
        strides=strides,
        dropout=0.2,
        norm='BATCH'
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CNN.apply(monai.networks.normal_init)
    CNN_model = CNN.to(device)
    CNN_model.load_state_dict(torch.load(model_file, map_location=device))
    CNN_model.eval()
      
    transform = Compose([
        LoadImaged(keys=["image"]),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)
    ])

    data_dicts = [{"image": t1w_gad}]
    infer_ds = Dataset(data=data_dicts, transform=transform)
    infer_loader = DataLoader(infer_ds, batch_size=1, shuffle=False)

    for infer_imgs in infer_loader:
        input_image = infer_imgs["image"].to(device)

        with torch.no_grad():
            input_image = torch.unsqueeze(input_image, dim=1)
            
            output_degad_img = sliding_window_inference(
                inputs=input_image, 
                roi_size=(32, 32, 32), 
                sw_batch_size=5,
                predictor=CNN, 
                overlap=0.15, 
                mode="gaussian",
                sigma_scale=0.25, 
                sw_device=device, 
                device='cpu', 
                progress=True
            )

        output_array = output_degad_img.squeeze().cpu().numpy()
        output_nifti = nib.Nifti1Image(output_array, affine=np.eye(4))
        nib.save(output_nifti, t1w_degad)

if __name__ == "__main__":
    
    model_file = get_model_file(
        model_dir=snakemake.input["model_dir"]
    )
    
    train_model(
        model_file,
        t1w_gad=snakemake.input["t1w_gad"],
        t1w_degad=snakemake.output["t1w_degad"]
    )