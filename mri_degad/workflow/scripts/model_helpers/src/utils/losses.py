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
    def __init__(self, net='alex', device='cpu'):
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
        raise ImportError("SSIM loss requires 'pytorch-msssim'. Please install it using 'pip install pytorch-msssim'")
    ssim_score = ssim(y_pred, y_true, data_range=max_val, size_average=True)
    return (1 - ssim_score) / 2.0

def ms_ssim_loss(y_pred, y_true, max_val=1.0):
    """
    Computes the multi-scale structural similarity (MS-SSIM) loss.
    Note: Requires the 'pytorch-msssim' package. `pip install pytorch-msssim`
    """
    if not _ssim_available:
        raise ImportError("MS-SSIM loss requires 'pytorch-msssim'. Please install it using 'pip install pytorch-msssim'")
    ms_ssim_score = ms_ssim(y_pred, y_true, data_range=max_val, size_average=True)
    return (1 - ms_ssim_score) / 2.0

def mae_loss(y_pred, y_true):
    """Computes the Mean Absolute Error (L1 loss)."""
    return F.l1_loss(y_pred, y_true)

def mse_loss(y_pred, y_true):
    """Computes the Mean Squared Error (L2 loss)."""
    return F.mse_loss(y_pred, y_true)

if __name__ == '__main__':
    print("--- Note: For SSIM/MS-SSIM functions, install with 'pip install pytorch-msssim' ---")
    print("--- Note: For LPIPS perceptual loss, install with 'pip install lpips' ---")
    
    # Create dummy tensors (B, C, H, W), assuming values are in [0, 1]
    dummy_pred = torch.rand(2, 1, 256, 256)
    dummy_true = torch.rand(2, 1, 256, 256)
    
    # --- Test Perceptual Loss (LPIPS) ---
    print("\n1. Testing Perceptual Loss (LPIPS)...")
    try:
        perceptual_loss_fn = PerceptualLoss()
        p_loss = perceptual_loss_fn(dummy_pred, dummy_true)
        print(f"   Perceptual Loss (LPIPS): {p_loss.item():.4f}")
    except Exception as e:
        print(f"   Could not compute Perceptual Loss. Ensure 'lpips' is installed.")
        print(f"   Error: {e}")

    # --- Test SSIM Loss ---
    print("\n2. Testing SSIM Loss...")
    try:
        s_loss = ssim_loss(dummy_pred, dummy_true)
        print(f"   SSIM Loss: {s_loss.item():.4f}")
    except ImportError as e:
        print(f"   {e}")

    # --- Test MS-SSIM Loss ---
    print("\n3. Testing MS-SSIM Loss...")
    try:
        ms_loss = ms_ssim_loss(dummy_pred, dummy_true)
        print(f"   MS-SSIM Loss: {ms_loss.item():.4f}")
    except ImportError as e:
        print(f"   {e}")

    # --- Test MAE Loss ---
    print("\n4. Testing MAE Loss...")
    m_loss = mae_loss(dummy_pred, dummy_true)
    print(f"   MAE Loss: {m_loss.item():.4f}")
    
    # --- Test MSE Loss ---
    print("\n5. Testing MSE Loss...")
    mse_val = mse_loss(dummy_pred, dummy_true)
    print(f"   MSE Loss: {mse_val.item():.4f}")
