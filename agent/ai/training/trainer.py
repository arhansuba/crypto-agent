"""
Training system implementation for AI crypto trading models.
Provides advanced training capabilities with monitoring, validation, and hyperparameter optimization.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
from datetime import datetime
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    epoch: int
    train_loss: float
    val_loss: float
    prediction_accuracy: float
    trend_accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    timestamp: datetime

class ModelTrainer:
    """
    Comprehensive training system for market prediction and risk analysis models.
    Includes advanced training features and performance monitoring.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        logger: logging.Logger,
        experiment_name: str
    ):
        self.model = model
        self.config = config
        self.logger = logger
        self.experiment_name = experiment_name
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Initialize training components
        self.criterion = self._setup_loss_function()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Initialize metrics tracking
        self.metrics_history: List[TrainingMetrics] = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Setup wandb monitoring
        self._initialize_wandb()
        
        self.logger.info(
            f"Trainer initialized on device: {self.device}, Experiment: {experiment_name}"
        )

    def _setup_loss_function(self) -> nn.Module:
        """Configure the loss function based on config settings"""
        loss_config = self.config['training']['loss']
        
        if loss_config['type'] == 'huber':
            return nn.HuberLoss(delta=loss_config.get('delta', 1.0))
        elif loss_config['type'] == 'mse':
            return nn.MSELoss()
        elif loss_config['type'] == 'custom':
            return self._create_custom_loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_config['type']}")

    def _create_custom_loss(self) -> nn.Module:
        """Create custom loss function for trading objectives"""
        class TradingLoss(nn.Module):
            def __init__(self, alpha: float = 0.5, beta: float = 0.3):
                super().__init__()
                self.alpha = alpha
                self.beta = beta
                self.mse = nn.MSELoss()
                
            def forward(
                self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                confidence: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
                # Basic prediction loss
                pred_loss = self.mse(predictions, targets)
                
                # Directional accuracy loss
                pred_diff = predictions[:, 1:] - predictions[:, :-1]
                target_diff = targets[:, 1:] - targets[:, :-1]
                direction_loss = -torch.mean(torch.sign(pred_diff) * torch.sign(target_diff))
                
                # Confidence-weighted loss if available
                if confidence is not None:
                    confidence_loss = -torch.mean(confidence * torch.abs(predictions - targets))
                    return pred_loss + self.alpha * direction_loss + self.beta * confidence_loss
                
                return pred_loss + self.alpha * direction_loss
        
        return TradingLoss(
            alpha=self.config['training']['loss'].get('alpha', 0.5),
            beta=self.config['training']['loss'].get('beta', 0.3)
        )

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Configure the optimizer"""
        optim_config = self.config['training']['optimizer']
        
        if optim_config['type'] == 'adamw':
            return AdamW(
                self.model.parameters(),
                lr=optim_config['learning_rate'],
                weight_decay=optim_config.get('weight_decay', 0.01)
            )
        else:
            return Adam(
                self.model.parameters(),
                lr=optim_config['learning_rate']
            )

    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Configure the learning rate scheduler"""
        scheduler_config = self.config['training']['scheduler']
        
        if scheduler_config['type'] == 'one_cycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=scheduler_config['max_lr'],
                epochs=self.config['training']['epochs'],
                steps_per_epoch=scheduler_config['steps_per_epoch']
            )
        elif scheduler_config['type'] == 'cosine':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=scheduler_config['T_0'],
                T_mult=scheduler_config.get('T_mult', 1)
            )
        return None

    def _initialize_wandb(self):
        """Initialize Weights & Biases monitoring"""
        try:
            wandb.init(
                project=self.config['monitoring']['project_name'],
                name=self.experiment_name,
                config=self.config
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {str(e)}")

    async def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: Optional[Path] = None
    ) -> List[TrainingMetrics]:
        """Execute the training loop"""
        try:
            for epoch in range(self.config['training']['epochs']):
                # Training phase
                self.model.train()
                train_metrics = await self._train_epoch(train_loader, epoch)
                
                # Validation phase
                self.model.eval()
                val_metrics = await self._validate_epoch(val_loader, epoch)
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Record metrics
                metrics = TrainingMetrics(
                    epoch=epoch,
                    train_loss=train_metrics['loss'],
                    val_loss=val_metrics['loss'],
                    prediction_accuracy=val_metrics['prediction_accuracy'],
                    trend_accuracy=val_metrics['trend_accuracy'],
                    sharpe_ratio=val_metrics['sharpe_ratio'],
                    max_drawdown=val_metrics['max_drawdown'],
                    timestamp=datetime.now()
                )
                
                self.metrics_history.append(metrics)
                
                # Log metrics
                self._log_metrics(metrics)
                
                # Save best model
                if metrics.val_loss < self.best_val_loss:
                    self.best_val_loss = metrics.val_loss
                    self.patience_counter = 0
                    if save_path:
                        self._save_model(save_path, metrics)
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config['training']['patience']:
                    self.logger.info("Early stopping triggered")
                    break
                
            return self.metrics_history
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    async def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Execute one training epoch"""
        total_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            features, targets = [b.to(self.device) for b in batch]
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features)
            
            # Calculate loss
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('clip_grad'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['clip_grad']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': total_loss / batch_count})
        
        return {
            'loss': total_loss / batch_count
        }

    async def _validate_epoch(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Execute validation phase"""
        total_loss = 0
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                features, targets = [b.to(self.device) for b in batch]
                
                # Forward pass
                predictions = self.model(features)
                loss = self.criterion(predictions, targets)
                
                # Collect predictions and targets
                predictions_list.append(predictions.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
                
                total_loss += loss.item()
        
        # Calculate performance metrics
        predictions = np.concatenate(predictions_list)
        targets = np.concatenate(targets_list)
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'prediction_accuracy': self._calculate_prediction_accuracy(predictions, targets),
            'trend_accuracy': self._calculate_trend_accuracy(predictions, targets),
            'sharpe_ratio': self._calculate_sharpe_ratio(predictions, targets),
            'max_drawdown': self._calculate_max_drawdown(predictions)
        }
        
        return metrics

    def _log_metrics(self, metrics: TrainingMetrics):
        """Log training metrics"""
        # Log to wandb
        try:
            wandb.log({
                'train_loss': metrics.train_loss,
                'val_loss': metrics.val_loss,
                'prediction_accuracy': metrics.prediction_accuracy,
                'trend_accuracy': metrics.trend_accuracy,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown
            })
        except Exception:
            pass
        
        # Log to console
        self.logger.info(
            f"Epoch {metrics.epoch}: "
            f"Train Loss: {metrics.train_loss:.4f}, "
            f"Val Loss: {metrics.val_loss:.4f}, "
            f"Pred Acc: {metrics.prediction_accuracy:.2%}, "
            f"Trend Acc: {metrics.trend_accuracy:.2%}"
        )

    def _save_model(self, save_path: Path, metrics: TrainingMetrics):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'metrics': metrics.__dict__,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, save_path)
            
            self.logger.info(f"Model checkpoint saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")

    @staticmethod
    def _calculate_prediction_accuracy(
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold: float = 0.05
    ) -> float:
        """Calculate prediction accuracy within threshold"""
        within_threshold = np.abs(predictions - targets) <= threshold * targets
        return np.mean(within_threshold)

    @staticmethod
    def _calculate_trend_accuracy(
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Calculate directional accuracy"""
        pred_direction = np.sign(predictions[:, 1:] - predictions[:, :-1])
        target_direction = np.sign(targets[:, 1:] - targets[:, :-1])
        return np.mean(pred_direction == target_direction)

    @staticmethod
    def _calculate_sharpe_ratio(
        predictions: np.ndarray,
        targets: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio of predictions"""
        returns = (predictions - targets) / targets
        excess_returns = returns - risk_free_rate / 252  # Daily adjustment
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

    @staticmethod
    def _calculate_max_drawdown(prices: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / peak
        return np.max(drawdown)