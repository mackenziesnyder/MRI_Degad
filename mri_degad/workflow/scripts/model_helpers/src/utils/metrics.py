import torch
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim, ms_ssim
    _ssim_available = True
except ImportError:
    _ssim_available = False

def psnr(y_pred, y_true, max_val=1.0):
    """Computes the Peak Signal-to-Noise Ratio metric."""
    mse = F.mse_loss(y_pred, y_true)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def mse_metric(y_pred, y_true):
    """Computes the Mean Squared Error metric."""
    return F.mse_loss(y_pred, y_true)

def mae_metric(y_pred, y_true):
    """Computes the Mean Absolute Error metric."""
    return F.l1_loss(y_pred, y_true)

def ssim_score(y_pred, y_true, max_val=1.0):
    """
    Computes the structural similarity (SSIM) score.
    Note: Requires the 'pytorch-msssim' package.
    """
    if not _ssim_available:
        raise ImportError("SSIM score requires 'pytorch-msssim'. Please install it using 'pip install pytorch-msssim'")
    return ssim(y_pred, y_true, data_range=max_val, size_average=True)

def ms_ssim_score(y_pred, y_true, max_val=1.0):
    """
    Computes the multi-scale structural similarity (MS-SSIM) score.
    Note: Requires the 'pytorch-msssim' package.
    """
    if not _ssim_available:
        raise ImportError("MS-SSIM score requires 'pytorch-msssim'. Please install it using 'pip install pytorch-msssim'")
    return ms_ssim(y_pred, y_true, data_range=max_val, size_average=True)


if __name__ == '__main__':
    print("--- Note: For SSIM/MS-SSIM functions, install with 'pip install pytorch-msssim' ---")
    print("--- Note: For LPIPS perceptual loss, install with 'pip install lpips' ---")
    
    # Create dummy tensors (B, C, H, W), assuming values are in [0, 1]
    dummy_pred = torch.rand(2, 1, 256, 256)
    dummy_true = torch.rand(2, 1, 256, 256)
    # --- Test PSNR Metric ---
    print("\n1. Testing PSNR Metric...")
    psnr_val = psnr(dummy_pred, dummy_true)
    print(f"   PSNR: {psnr_val.item():.4f} dB")
    
    # --- Test SSIM Score Metric ---
    print("\n2. Testing SSIM Score...")
    try:
        ssim_val = ssim_score(dummy_pred, dummy_true)
        print(f"   SSIM Score: {ssim_val.item():.4f}")
    except ImportError as e:
        print(f"   {e}")

    # --- Test MS-SSIM Score Metric ---
    print("\n3. Testing MS-SSIM Score...")
    try:
        ms_ssim_val = ms_ssim_score(dummy_pred, dummy_true)
        print(f"   MS-SSIM Score: {ms_ssim_val.item():.4f}")
    except ImportError as e:
        print(f"   {e}") 