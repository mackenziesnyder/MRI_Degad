import torch
import pytorch_lightning as pl
from model_helpers.losses import PerceptualLoss, ssim_loss
from model_helpers.wat_stacked_unets import WATStackedUNets

class Model(pl.LightningModule):
    def __init__(self, device):
        super().__init__()
        self.save_hyperparameters()
        self.device_ = device
        self.learning_rate = 0.001
        self.loss_functions = self._get_loss_functions(device)
        self.model = self._get_model(device)
        self.automatic_optimization = True
        
        # Initialize with default weights - will be updated after distributed setup
        self.weights = torch.ones(len(self.loss_functions), dtype=torch.float32, device=device)
        self.biases = torch.zeros(len(self.loss_functions), dtype=torch.float32, device=device)
        
        # Flag to track if adaptive loss has been computed
        self.adaptive_loss_computed = False

    def _get_model(self, device):
        return WATStackedUNets().to(device)

    def _get_loss_functions(self, device):
        losses = [ssim_loss, PerceptualLoss(device=device)]
        return losses
