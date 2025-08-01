import os
import sys
import math
import argparse
import logging
from datetime import datetime
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
warnings.filterwarnings("ignore", category=UserWarning, module="torchio.data.image")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed.algorithms.ddp_comm_hooks")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.trainer.connectors.logger_connector.result")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.trainer.connectors.data_connector")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_fabric.plugins.environments.slurm")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.trainer.connectors.logger_connector.logger_connector")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.distributed_c10d")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.callbacks.model_checkpoint")

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from src.data.dataloader import NiftiSliceDataset
from src.utils.losses import PerceptualLoss, ssim_loss, ms_ssim_loss, mae_loss, mse_loss
from src.utils.metrics import psnr, ssim_score, ms_ssim_score, mse_metric, mae_metric
from src.utils.adaptive_losses import AdaMultiLossesNormTorch
from src.models.stacked_unets import StackedUNets
from src.models.wat_stacked_unets import WATStackedUNets
from src.models.wat_unet import MultiInputWATUNet
from src.models.unet_vanilla import UNetVanilla


def setup_logging(output_path, log_level=logging.INFO):
    """
    Set up logging to both console and file.
    
    Args:
        output_path (str): Path to the output directory
        log_level: Logging level (default: logging.INFO)
    """
    import pytorch_lightning as pl
    # Only initialize file logging on rank 0 (main process)
    rank = 0
    if hasattr(pl.utilities, 'rank_zero_only'):
        try:
            rank = pl.utilities.rank_zero_only.rank
        except Exception:
            rank = 0
    if rank != 0:
        return logging.getLogger(__name__)
    # Create logs directory in output path
    logs_dir = os.path.join(output_path, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(logs_dir, f'training_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # ensures config is always applied
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger


class DataModule(pl.LightningDataModule):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.batch_size = config['batch_size']
        self.data_loader_config = config['data_loader']
        self.dataset_path = config.get('dataset', config.get('data_path', 'data'))
        
        # Setup logging
        output_path = config.get('output_path', 'outputs')
        self.logger = setup_logging(output_path)

    def setup(self, stage=None):
        # Check if dataset directory exists
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Dataset directory does not exist: {self.dataset_path}")
        
        all_subjects = [d for d in os.listdir(self.dataset_path) if d.startswith('sub')]
        split_ratio = self.data_loader_config.get('split_ratio', [0.7, 0.2, 0.1])
        
        total_subjects = len(all_subjects)
        
        # Log dataset information instead of printing
        self.logger.info(f"Dataset path: {self.dataset_path}")
        self.logger.info(f"All subjects found: {all_subjects}")
        self.logger.info(f"Total subjects: {total_subjects}")
        
        if total_subjects < 3:
            # For very small datasets, use all data for training
            self.logger.warning(f"Only {total_subjects} subjects found. Using all for training.")
            train_size = total_subjects
            val_size = 0
            test_size = 0
        else:
            # Calculate initial split sizes
            train_size = max(1, int(split_ratio[0] * total_subjects))
            val_size = max(1, int(split_ratio[1] * total_subjects))
            test_size = max(1, int(split_ratio[2] * total_subjects))
            
            # Adjust if total exceeds available subjects
            total_allocated = train_size + val_size + test_size
            if total_allocated > total_subjects:
                # Redistribute excess subjects
                excess = total_allocated - total_subjects
                if excess == 1:
                    # Reduce the largest split by 1
                    if train_size >= val_size and train_size >= test_size:
                        train_size -= 1
                    elif val_size >= test_size:
                        val_size -= 1
                    else:
                        test_size -= 1
                elif excess == 2:
                    # Reduce the two largest splits by 1 each
                    sizes = [(train_size, 'train'), (val_size, 'val'), (test_size, 'test')]
                    sizes.sort(reverse=True)
                    if sizes[0][1] == 'train':
                        train_size -= 1
                    elif sizes[0][1] == 'val':
                        val_size -= 1
                    else:
                        test_size -= 1
                    
                    if sizes[1][1] == 'train':
                        train_size -= 1
                    elif sizes[1][1] == 'val':
                        val_size -= 1
                    else:
                        test_size -= 1
        
        self.logger.info(f"Total subjects: {total_subjects}")
        self.logger.info(f"Split ratio: {split_ratio}")
        self.logger.info(f"Split sizes: Train={train_size}, Val={val_size}, Test={test_size}")
        
        train_subjects = all_subjects[:train_size]
        val_subjects = all_subjects[train_size:train_size + val_size]
        test_subjects = all_subjects[train_size + val_size:]
        
        self.logger.info(f"Train subjects: {train_subjects}")
        self.logger.info(f"Val subjects: {val_subjects}")
        self.logger.info(f"Test subjects: {test_subjects}")
        
        if stage == 'fit' or stage is None:
            # Get preload probability from config, with different defaults for train/val
            train_preload_prob = self.data_loader_config.get('train_preload_probability', 0.5)  # Lower for training
            val_preload_prob = self.data_loader_config.get('val_preload_probability', 1.0)      # Higher for validation
            
            self.train_dataset = NiftiSliceDataset(
                data_path=self.dataset_path,
                subjects=train_subjects,
                view=self.data_loader_config.get('view', 'axial').lower(),
                data_id=self.data_loader_config.get('data_id', 'motion').lower(),
                crop=self.data_loader_config.get('crop', True),
                enable_SAP=self.data_loader_config.get('enable_SAP', True),
                preload_probability=train_preload_prob
            )
            self.val_dataset = NiftiSliceDataset(
                data_path=self.dataset_path,
                subjects=val_subjects,
                view=self.data_loader_config.get('view', 'axial').lower(),
                data_id=self.data_loader_config.get('data_id', 'motion').lower(),
                crop=self.data_loader_config.get('crop', True),
                enable_SAP=self.data_loader_config.get('enable_SAP', True),
                preload_probability=val_preload_prob
            )
            if len(self.train_dataset) == 0 or len(self.val_dataset) == 0:
                raise ValueError("Train or validation dataset is empty.")
            
            # Additional debugging for empty datasets
            if len(self.train_dataset) == 0:
                self.logger.error(f"WARNING: Training dataset is empty!")
                self.logger.error(f"  Data path: {self.dataset_path}")
                self.logger.error(f"  Train subjects: {train_subjects}")
                self.logger.error(f"  View: {self.data_loader_config.get('view', 'axial').lower()}")
                self.logger.error(f"  Data ID: {self.data_loader_config.get('data_id', 'motion').lower()}")
                self.logger.error(f"  Crop: {self.data_loader_config.get('crop', True)}")
                self.logger.error(f"  Enable SAP: {self.data_loader_config.get('enable_SAP', True)}")

        if stage == 'test' or stage is None:
            # For testing, we want to load all volumes to ensure complete evaluation
            test_preload_prob = self.data_loader_config.get('test_preload_probability', 1.0)
            
            try:
                self.test_dataset = NiftiSliceDataset(
                    data_path=self.dataset_path,
                    subjects=test_subjects,
                    view=self.data_loader_config.get('view', 'axial').lower(),
                    data_id=self.data_loader_config.get('data_id', 'motion').lower(),
                    crop=self.data_loader_config.get('crop', True),
                    enable_SAP=self.data_loader_config.get('enable_SAP', True),
                    preload_probability=test_preload_prob
                )
                if len(self.test_dataset) == 0:
                    self.logger.warning("Test dataset is empty. Test evaluation will be skipped.")
                    self.test_dataset = None
            except Exception as e:
                self.logger.warning(f"Failed to create test dataset: {e}. Test evaluation will be skipped.")
                self.test_dataset = None
        
        # Conditional logging based on the stage
        if stage == 'fit':
            self.logger.info(f"DataLoaders initialized for 'fit' stage. Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
        elif stage == 'test':
            self.logger.info(f"DataLoader initialized for 'test' stage. Test: {len(self.test_dataset) if self.test_dataset else 'Skipped'}")
        elif stage is None:
            self.logger.info(f"DataLoaders initialized successfully. Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset) if self.test_dataset else 'Skipped'}")

    def train_dataloader(self):
        if len(self.train_dataset) == 0:
            raise ValueError(f"Training dataset is empty. Check data path: {self.dataset_path}")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                         num_workers=4, pin_memory=self.device.type == 'cuda', 
                         persistent_workers=True)
    
    def val_dataloader(self):
        if len(self.val_dataset) == 0:
            raise ValueError(f"Validation dataset is empty. Check data path: {self.dataset_path}")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, 
                         num_workers=1, pin_memory=self.device.type == 'cuda',
                         persistent_workers=False)  # No persistent workers for validation
    
    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset is not available. Check logs for details.")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, 
                         num_workers=1, pin_memory=self.device.type == 'cuda',
                         persistent_workers=False)  # No persistent workers for testing

class Model(pl.LightningModule):
    def __init__(self, config, device):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.device_ = device
        self.learning_rate = config['learning_rate']
        self.loss_functions = self._get_loss_functions(config['loss_functions'], device)
        self.model = self._get_model(config['model_architecture'], device)
        self.automatic_optimization = True
        
        # Setup logging for the model (use custom_logger to avoid conflict with Lightning's logger)
        output_path = config.get('output_path', 'outputs')
        self.custom_logger = setup_logging(output_path)
        
        # Initialize with default weights - will be updated after distributed setup
        self.weights = torch.ones(len(self.loss_functions), dtype=torch.float32, device=device)
        self.biases = torch.zeros(len(self.loss_functions), dtype=torch.float32, device=device)
        
        # Flag to track if adaptive loss has been computed
        self.adaptive_loss_computed = False

    def setup(self, stage=None):
        """Called after the model is distributed across GPUs"""
        super().setup(stage)
        
        # Only compute adaptive loss once after distributed setup
        if not self.adaptive_loss_computed and len(self.loss_functions) > 1 and not self.config.get('skip_ada_multi_losses_norm', False):
            self._compute_adaptive_loss_weights()
            self.adaptive_loss_computed = True

    def _compute_adaptive_loss_weights(self):
        """Compute adaptive loss weights and biases, handling distributed training properly"""
        # Check if we're in a distributed setting
        if torch.distributed.is_initialized():
            # In DDP, only compute on main process (rank 0)
            if self.global_rank == 0:
                self.custom_logger.info("Computing adaptive loss weights and biases on main process...")
                # Use a dummy dataloader for AdaMultiLossesNormTorch
                dm = DataModule(self.config, self.device_)
                dm.setup(stage='fit')  # Only set up training stage for adaptive loss computation
                
                # Check if datasets are not empty before computing adaptive losses
                if len(dm.train_dataset) == 0:
                    self.custom_logger.warning("Training dataset is empty. Skipping adaptive loss computation.")
                    weights = [1.0] * len(self.loss_functions)
                    biases = [0.0] * len(self.loss_functions)
                else:
                    ada_loss = AdaMultiLossesNormTorch()
                    losses = ada_loss.compute_losses(dm.train_dataloader(), self.config['batch_size'], losses=self.loss_functions, device=self.device_)
                    n_loss, weights, biases = ada_loss.compute_normalized_weights_and_biases(*losses)
                    
                    self.custom_logger.info(f"Adaptive loss weights computed: {weights}")
                    self.custom_logger.info(f"Adaptive loss biases computed: {biases}")
                    
                    # Plot loss distributions if enabled
                    if self.config.get("enable_loss_plotting", True):
                        self.custom_logger.info("Creating loss distribution plots...")
                        loss_names = self.config['loss_functions']
                        plots_dir = os.path.join(self.config['output_path'], "plots")
                        ada_loss.plot_loss_distributions(
                            losses=losses,
                            weights=weights,
                            biases=biases,
                            loss_names=loss_names,
                            save_path=plots_dir,
                            save_individual_plots=self.config.get("plotting", {}).get("save_individual_plots", True),
                            plot_dpi=self.config.get("plotting", {}).get("plot_dpi", 300),
                            plot_format=self.config.get("plotting", {}).get("plot_format", "png")
                        )
                        self.custom_logger.info(f"Loss distribution plots saved to: {plots_dir}")
            else:
                # Non-main processes use default weights initially
                weights = [1.0] * len(self.loss_functions)
                biases = [0.0] * len(self.loss_functions)
            
            # Broadcast weights and biases from main process to all processes
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device_)
            biases_tensor = torch.tensor(biases, dtype=torch.float32, device=self.device_)
            
            torch.distributed.broadcast(weights_tensor, src=0)
            torch.distributed.broadcast(biases_tensor, src=0)
            
            self.weights = weights_tensor
            self.biases = biases_tensor
            
            # Log on all processes after broadcast
            if self.global_rank == 0:
                self.custom_logger.info("Adaptive loss weights and biases broadcasted to all processes")
        else:
            # Single GPU or CPU training
            self.custom_logger.info("Computing adaptive loss weights and biases...")
            # Use a dummy dataloader for AdaMultiLossesNormTorch
            dm = DataModule(self.config, self.device_)
            dm.setup(stage='fit')  # Only set up training stage for adaptive loss computation
            
            # Check if datasets are not empty before computing adaptive losses
            if len(dm.train_dataset) == 0:
                self.custom_logger.warning("Training dataset is empty. Skipping adaptive loss computation.")
                self.weights = torch.ones(len(self.loss_functions), dtype=torch.float32, device=self.device_)
                self.biases = torch.zeros(len(self.loss_functions), dtype=torch.float32, device=self.device_)
            else:
                ada_loss = AdaMultiLossesNormTorch()
                losses = ada_loss.compute_losses(dm.train_dataloader(), self.config['batch_size'], losses=self.loss_functions, device=self.device_)
                n_loss, weights, biases = ada_loss.compute_normalized_weights_and_biases(*losses)
                self.weights = torch.tensor(weights, dtype=torch.float32, device=self.device_)
                self.biases = torch.tensor(biases, dtype=torch.float32, device=self.device_)
                
                self.custom_logger.info(f"Adaptive loss weights computed: {weights}")
                self.custom_logger.info(f"Adaptive loss biases computed: {biases}")
                
                # Plot loss distributions if enabled
                if self.config.get("enable_loss_plotting", True):
                    self.custom_logger.info("Creating loss distribution plots...")
                    loss_names = self.config['loss_functions']
                    plots_dir = os.path.join(self.config['output_path'], "plots")
                    ada_loss.plot_loss_distributions(
                        losses=losses,
                        weights=weights,
                        biases=biases,
                        loss_names=loss_names,
                        save_path=plots_dir,
                        save_individual_plots=self.config.get("plotting", {}).get("save_individual_plots", True),
                        plot_dpi=self.config.get("plotting", {}).get("plot_dpi", 300),
                        plot_format=self.config.get("plotting", {}).get("plot_format", "png")
                    )
                    self.custom_logger.info(f"Loss distribution plots saved to: {plots_dir}")

    def _get_model(self, name, device):
        if name == "stacked_unets":
            return StackedUNets().to(device)
        elif name == "wat_stacked_unets":
            return WATStackedUNets().to(device)
        elif name == "wat_unet":
            return MultiInputWATUNet().to(device)
        elif name == "vanilla_unet":
            return UNetVanilla().to(device)
        else:
            raise ValueError(f"Unknown model: {name}")

    def _get_loss_functions(self, loss_names, device):
        losses = []
        for name in loss_names:
            if name == "ssim_loss":
                losses.append(ssim_loss)
            elif name == "ms_ssim_loss":
                losses.append(ms_ssim_loss)
            elif name == "perceptual_loss":
                losses.append(PerceptualLoss(device=device))
            elif name == "mae_loss":
                losses.append(mae_loss)
            elif name == "mse_loss":
                losses.append(mse_loss)
            else:
                raise ValueError(f"Unknown loss: {name}")
        return losses

    def forward(self, x1, x2, x3):
        return self.model(x1, x2, x3)

    def training_step(self, batch, batch_idx):
        motion_data, target = batch
        if isinstance(motion_data, (list, tuple)):
            x1, x2, x3 = [m.to(self.device_) for m in motion_data]
        else:
            x1 = x2 = x3 = motion_data.to(self.device_)
        target = target.to(self.device_)
        output = self(x1, x2, x3)
        device = output.device
        weights = self.weights.to(device)
        biases = self.biases.to(device)
        loss = 0
        for i, loss_fn in enumerate(self.loss_functions):
            loss_val = loss_fn(output, target)
            loss += weights[i] * loss_val + biases[i]
        loss = loss / len(self.loss_functions)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        motion_data, target = batch
        if isinstance(motion_data, (list, tuple)):
            x1, x2, x3 = [m.to(self.device_) for m in motion_data]
        else:
            x1 = x2 = x3 = motion_data.to(self.device_)
        target = target.to(self.device_)
        output = self(x1, x2, x3)
        device = output.device
        weights = self.weights.to(device)
        biases = self.biases.to(device)
        loss = 0
        for i, loss_fn in enumerate(self.loss_functions):
            loss_val = loss_fn(output, target)
            loss += weights[i] * loss_val + biases[i]
        loss = loss / len(self.loss_functions)
        psnr_val = psnr(output, target)
        ssim_val = ssim_score(output, target)
        ms_ssim_val = ms_ssim_score(output, target)
        mse_val = mse_metric(output, target)
        mae_val = mae_metric(output, target)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_psnr', psnr_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_ssim', ssim_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_ms_ssim', ms_ssim_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mse', mse_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae_val, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_psnr': psnr_val, 'val_ssim': ssim_val, 'val_ms_ssim': ms_ssim_val, 'val_mse': mse_val, 'val_mae': mae_val}

    def test_step(self, batch, batch_idx):
        motion_data, target = batch
        if isinstance(motion_data, (list, tuple)):
            x1, x2, x3 = [m.to(self.device_) for m in motion_data]
        else:
            x1 = x2 = x3 = motion_data.to(self.device_)
        target = target.to(self.device_)
        output = self(x1, x2, x3)
        device = output.device
        weights = self.weights.to(device)
        biases = self.biases.to(device)
        loss = 0
        for i, loss_fn in enumerate(self.loss_functions):
            loss_val = loss_fn(output, target)
            loss += weights[i] * loss_val + biases[i]
        loss = loss / len(self.loss_functions)
        psnr_val = psnr(output, target)
        ssim_val = ssim_score(output, target)
        ms_ssim_val = ms_ssim_score(output, target)
        mse_val = mse_metric(output, target)
        mae_val = mae_metric(output, target)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_psnr', psnr_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_ssim', ssim_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_ms_ssim', ms_ssim_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_mse', mse_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_mae', mae_val, on_step=False, on_epoch=True, prog_bar=True)
        return {'test_loss': loss, 'test_psnr': psnr_val, 'test_ssim': ssim_val, 'test_ms_ssim': ms_ssim_val, 'test_mse': mse_val, 'test_mae': mae_val}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['max_epochs']
        )
        return [optimizer], [scheduler]

def main(config_path, dataset=None, devices=None, accelerator=None):
    # Configure torch to avoid warnings
    torch.set_float32_matmul_precision('medium')
    
    with open(config_path) as f:
        config = json.load(f)
    if dataset:
        config['dataset'] = dataset
    
    # Save a copy of the config as output_path.json in output_path
    os.makedirs(config['output_path'], exist_ok=True)
    output_config_path = os.path.join(config['output_path'], 'config.json')
    with open(output_config_path, 'w') as f_out:
        json.dump(config, f_out, indent=2)
    
    # Setup logging for main function
    logger = setup_logging(config['output_path'])
    logger.info("=" * 60)
    logger.info("Starting MRI Motion Correction Training")
    logger.info("=" * 60)
    logger.info(f"Configuration file: {config_path}")
    logger.info(f"Output path: {config['output_path']}")
    logger.info(f"Model architecture: {config['model_architecture']}")
    logger.info(f"Learning rate: {config['learning_rate']}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Loss functions: {config['loss_functions']}")
    
    # Handle checkpoint loading
    checkpoint_path = config.get('checkpoint_path')
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading from checkpoint: {checkpoint_path}")
            logger.info(f"Checkpoint file size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")
        else:
            logger.warning(f"Checkpoint path specified but file does not exist: {checkpoint_path}")
            logger.info("Starting training from scratch")
            checkpoint_path = None
    else:
        logger.info("No checkpoint specified. Starting training from scratch")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    pl.seed_everything(42)
    logger.info("Random seed set to 42 for reproducibility")
    
    try:
        datamodule = DataModule(config, device)
        
        # Initialize model
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading model from checkpoint: {checkpoint_path}")
            model = Model.load_from_checkpoint(checkpoint_path, config=config, device=device)
            logger.info("Model loaded successfully from checkpoint")
        else:
            logger.info("Initializing new model")
            model = Model(config, device)
        
        logger.info("Model and DataModule initialized successfully")
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=config['output_path'],
            filename=config['model_architecture'] + '-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            monitor='val_loss',
            mode='min',
            save_last=True
        )
        
        trainer_kwargs = dict(
            max_epochs=config['max_epochs'],
            callbacks=[checkpoint_callback],
            log_every_n_steps=10,
            val_check_interval=1.0,
            check_val_every_n_epoch=1,
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=True,
            default_root_dir=config['output_path'],
            logger=False  # Disable default logger to avoid TensorBoard warning
        )
        if devices is not None:
            trainer_kwargs['devices'] = devices
        else:
            trainer_kwargs['devices'] = 'auto'
        if accelerator is not None:
            trainer_kwargs['accelerator'] = accelerator
        else:
            trainer_kwargs['accelerator'] = 'auto'
        trainer = pl.Trainer(**trainer_kwargs)
        
        logger.info(f"Trainer configured with max_epochs={config['max_epochs']}")
        logger.info(f"Checkpoint callback configured to save best model")
        
        # Log model summary
        logger.info("Model Summary:")
        logger.info(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        logger.info("Starting training...")
        
        # Start training with or without checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
            trainer.fit(model, datamodule, ckpt_path=checkpoint_path)
        else:
            logger.info("Starting training from scratch")
            trainer.fit(model, datamodule)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best model saved at: {checkpoint_callback.best_model_path}")
        logger.info(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
        
        # Always attempt test; Lightning will call setup(stage='test')
        logger.info("Running test evaluation...")
        try:
            trainer.test(model, datamodule)
            logger.info("Test evaluation completed successfully!")
        except Exception as e:
            logger.error(f"Test evaluation failed: {e}")
        
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error("Training failed!")
        raise

if __name__ == "__main__":
    import sys
    main(config_path=sys.argv[1]) 
