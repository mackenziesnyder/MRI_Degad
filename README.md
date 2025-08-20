# MRI-Degad: Conversion of gadolinium-enhanced T1-weighted MRI to non-gadolinium T1-weighted scans
[![Documentation Status](https://readthedocs.org/projects/mri_degad/badge/?version=latest)](https://mri-degad.readthedocs.io/en/stable/?badge=stable)
![Python3](https://img.shields.io/badge/python-_3.10_|_3.11_|_3.12-blue.svg)
![Tests](https://github.com/mackenziesnyder/MRI_Degad/actions/workflows/lint_and_test.yml/badge.svg?branch=main)

AIMS Lab Research Team at the Robarts Research Institute - 2024-2025

![Alt text](https://raw.githubusercontent.com/mackenziesnyder/mri_degad/master/docs/images/mri-degad-image.webp)

*This package is under active development. It should be stable and reproducible, but please let any of the active contributing members know if there are any bugs or unusual behaviour.*

This Python package is a program utilizing an adapted wat-stacked U-Net built off of the standard [U-Net](https://arxiv.org/abs/1505.04597) (Ronneberger et al. 2015) machine learning model. It is based on Snakemake and SnakeBIDS workflow management tools, trained on a database containing gadolinium-enhanced T1-weighted MRI and non-gadolinium T1-weighted scans. MRI-Degad outputs a sythetic T1w non-contrast scan, and a mask of the brain vasculature. It is currently in development phase and the user is highly advised to get familiar with the above mentioned workflow managaments tools and read docstrings and relevant documentation before using this software.

## Workflow

A brief summary of the workflow can be found below along with its Directed Acyclic Graph (DAG) (see documentation for a detailed summary):

![Alt text](https://raw.githubusercontent.com/mackenziesnyder/mri_degad/master/docs/images/dag.svg)

1. Preprocess input NIfTI T1w gadolinium enhanced files (n4 bias correction, isotropic 1mm voxel resampling, min/max normalization)
2. Download and apply the Degad model in the axial, sagittal, and coronal direction
3. Fuse the image 
4. Register the degad fused image to the gad image
5. Extract the vasculature mask

### **Full documentation:** [here](https://mri-degad.readthedocs.io/en/latest/)

## Revalent Papers: 
* Ogunsanya F, Taha A, Gilmore G, Kai J, Kuehn T, Thurairajah A, Tenorio MC, Khan AR, Lau JC.  
  [MRI-degad: toward accurate conversion of gadolinium-enhanced T1w MRIs to non-contrast-enhanced scans using CNNs](https://doi.org/10.1007/s11548-024-03186-z).  
  *International Journal of Computer Assisted Radiology and Surgery*. 2024 Jul;19(7):1469-1472. doi: 10.1007/s11548-024-03186-z. Epub 2024 Jun 1. PMID: 38822981.


## Questions, Issues, Suggestions, and Other Feedback
Please reach out if you have any questions, suggestions, or other feedback related to this software—either through email (m25snyde@uwaterloo.ca) or the discussions page. Larger issues or feature requests can be posted and tracked via the issues page.