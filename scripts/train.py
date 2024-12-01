"""
Training script for AI models.
Handles data preparation, model training, and performance evaluation.
"""

import asyncio
import argparse
from asyncio.log import logger
import logging
from pathlib import Path
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader
from datetime import datetime

from src.ai.models.market_predictor import MarketPredictor
from src.ai.models.risk_analyzer import RiskAnalyzer
from src.ai.training.dataset import MarketDataManager
from src.ai.training.trainer import ModelTrainer
from src.utils.logger import CryptoAgentLogger
from src.utils.config import ConfigManager

async def prepare_training_data(
    config: Dict,
    data_manager: MarketDataManager
) -> Tuple[DataLoader, DataLoader]:
    """Prepare training and validation datasets"""
    try:
        # Set data parameters
        start_date = datetime.strptime(
            config['training']['start_date'],
            '%Y-%m-%d'
        )
        end_date = datetime.strptime(
            config['training']['end_date'],
            '%Y-%m-%d'
        )
        
        # Create datasets
        train_loader, val_loader = await data_manager.create_dataset(
            symbol=config['training']['symbol'],
            start_time=start_date,
            end_time=end_date,
            sequence_length=config['model']['sequence_length'],
            prediction_horizon=config['model']['prediction_horizon'],
            batch_size=config['training']['batch_size']
        )
        
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        raise

async def train_models(
    config: Dict,
    logger: logging.Logger,
    train_loader: DataLoader,
    val_loader: DataLoader
):
    """Train market prediction and risk analysis models"""
    try:
        # Initialize market predictor
        market_predictor = MarketPredictor(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            dropout=config['model']['dropout'],
            prediction_horizon=config['model']['prediction_horizon']
        )
        
        # Initialize risk analyzer
        risk_analyzer = RiskAnalyzer(
            input_features=config['risk_model']['input_features'],
            hidden_dim=config['risk_model']['hidden_dim'],
            num_layers=config['risk_model']['num_layers']
        )
        
        # Initialize trainers
        market_trainer = ModelTrainer(
            market_predictor,
            config,
            logger,
            "market_predictor"
        )
        
        risk_trainer = ModelTrainer(
            risk_analyzer,
            config,
            logger,
            "risk_analyzer"
        )
        
        # Train models
        await market_trainer.train(train_loader, val_loader)
        await risk_trainer.train(train_loader, val_loader)
        
        # Save models
        save_path = Path(config['training']['save_path'])
        torch.save(
            market_predictor.state_dict(),
            save_path / 'market_predictor.pth'
        )
        torch.save(
            risk_analyzer.state_dict(),
            save_path / 'risk_analyzer.pth'
        )
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

async def main(config_path: str):
    """Main training function"""
    # Load configuration
    config_manager = ConfigManager(config_path)
    config = config_manager.get_all()
    
    # Setup logging
    logger = CryptoAgentLogger(config)
    
    try:
        # Initialize data manager
        data_manager = MarketDataManager(config, logger)
        
        # Prepare data
        logger.info("Preparing training data...")
        train_loader, val_loader = await prepare_training_data(
            config,
            data_manager
        )
        
        # Train models
        logger.info("Starting model training...")
        await train_models(config, logger, train_loader, val_loader)
        
    except Exception as e:
        logger.error(f"Training script failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    args = parser.parse_args()
    
    asyncio.run(main(args.config))
