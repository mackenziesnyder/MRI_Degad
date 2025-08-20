import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

try:
    from pytorch_msssim import ssim, ms_ssim

    _ssim_available = True
except ImportError:
    _ssim_available = False


class PerceptualLoss(nn.Module):
    """
    Calculates the perceptual loss using LPIPS (Learned Perceptual Image Patch Similarity).
    This is the state-of-the-art perceptual loss function.
    """

    def __init__(self, net="alex", device="cpu"):
        super(PerceptualLoss, self).__init__()
        self.loss_fn = lpips.LPIPS(net=net, verbose=False).to(device)

    def forward(self, y_pred, y_true):
        """
        Computes the perceptual loss.

        Args:
            y_pred (torch.Tensor): The predicted image tensor (B, C, H, W).
            y_true (torch.Tensor): The ground truth image tensor (B, C, H, W).
        """
        # Move loss_fn to the same device as y_pred for DDP/multi-GPU
        self.loss_fn = self.loss_fn.to(y_pred.device)
        return self.loss_fn(y_pred, y_true).mean()


def ssim_loss(y_pred, y_true, max_val=1.0):
    """
    Computes the structural similarity (SSIM) loss.
    Note: Requires the 'pytorch-msssim' package. `pip install pytorch-msssim`
    """
    if not _ssim_available:
        raise ImportError(
            "SSIM loss requires 'pytorch-msssim'. Please install it using 'pip install pytorch-msssim'"
        )
    ssim_score = ssim(y_pred, y_true, data_range=max_val, size_average=True)
    return (1 - ssim_score) / 2.0
