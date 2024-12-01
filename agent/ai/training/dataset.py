"""
Dataset handling implementation for AI crypto trading models.
Provides data loading, preprocessing, and augmentation capabilities for model training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging
from pathlib import Path
import aiohttp
import asyncio
from dataclasses import dataclass

@dataclass
class MarketDataPoint:
    """Represents a single market data point"""
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    trades_count: int
    quote_volume: float
    taker_buy_volume: float
    taker_sell_volume: float

class CryptoDataset(Dataset):
    """Base dataset class for cryptocurrency market data"""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        sequence_length: int,
        prediction_horizon: int,
        feature_columns: List[str],
        target_column: str,
        transform: Optional[object] = None,
        train_mode: bool = True
    ):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.transform = transform
        self.train_mode = train_mode
        
        self.data = self._load_data()
        self.sequences = self._prepare_sequences()

    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess market data"""
        try:
            # Load raw data
            df = pd.read_parquet(self.data_path)
            
            # Ensure datetime index
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Calculate additional features
            df['returns'] = df[self.target_column].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['price_ma'] = df[self.target_column].rolling(window=20).mean()
            
            # Drop rows with NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {str(e)}")

    def _prepare_sequences(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Prepare sequences for training or inference"""
        sequences = []
        
        for i in range(len(self.data) - self.sequence_length - self.prediction_horizon + 1):
            # Extract feature sequence
            feature_seq = self.data[self.feature_columns].iloc[i:i + self.sequence_length].values
            
            # Extract target sequence
            if self.train_mode:
                target_seq = self.data[self.target_column].iloc[
                    i + self.sequence_length:i + self.sequence_length + self.prediction_horizon
                ].values
                sequences.append((feature_seq, target_seq))
            else:
                sequences.append(feature_seq)
        
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.train_mode:
            features, target = self.sequences[idx]
            
            if self.transform:
                features = self.transform(features)
                target = self.transform(target.reshape(-1, 1)).squeeze()
            
            return torch.FloatTensor(features), torch.FloatTensor(target)
        else:
            features = self.sequences[idx]
            
            if self.transform:
                features = self.transform(features)
            
            return torch.FloatTensor(features)

class MarketDataManager:
    """Manages market data collection, preprocessing, and dataset creation"""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Initialize data storage
        self.data_path = Path(config['data']['storage_path'])
        self.data_path.mkdir(parents=True, exist_ok=True)

    async def fetch_market_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = '1h'
    ) -> pd.DataFrame:
        """Fetch market data from exchange API"""
        try:
            async with aiohttp.ClientSession() as session:
                # Construct API endpoint URL
                base_url = self.config['data']['api_endpoint']
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': int(start_time.timestamp() * 1000),
                    'endTime': int(end_time.timestamp() * 1000),
                    'limit': 1000
                }
                
                data_points = []
                current_time = start_time
                
                while current_time < end_time:
                    params['startTime'] = int(current_time.timestamp() * 1000)
                    
                    async with session.get(base_url, params=params) as response:
                        if response.status != 200:
                            raise RuntimeError(f"API request failed: {await response.text()}")
                        
                        batch_data = await response.json()
                        
                        for point in batch_data:
                            data_points.append(MarketDataPoint(
                                timestamp=datetime.fromtimestamp(point[0] / 1000),
                                open_price=float(point[1]),
                                high_price=float(point[2]),
                                low_price=float(point[3]),
                                close_price=float(point[4]),
                                volume=float(point[5]),
                                trades_count=int(point[8]),
                                quote_volume=float(point[7]),
                                taker_buy_volume=float(point[9]),
                                taker_sell_volume=float(point[10])
                            ))
                        
                        if not batch_data:
                            break
                        
                        current_time = data_points[-1].timestamp
                        await asyncio.sleep(0.1)  # Rate limiting
                
                # Convert to DataFrame
                df = pd.DataFrame([vars(p) for p in data_points])
                df.set_index('timestamp', inplace=True)
                return df
                
        except Exception as e:
            self.logger.error(f"Failed to fetch market data: {str(e)}")
            raise

    def preprocess_data(
        self,
        df: pd.DataFrame,
        scaler_key: str = 'default'
    ) -> Tuple[pd.DataFrame, StandardScaler]:
        """Preprocess market data for model training"""
        try:
            # Calculate technical indicators
            df['rsi'] = self._calculate_rsi(df['close_price'])
            df['macd'], df['macd_signal'] = self._calculate_macd(df['close_price'])
            df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['close_price'])
            
            # Calculate price momentum
            df['momentum'] = df['close_price'].pct_change(periods=10)
            
            # Calculate volume indicators
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_std'] = df['volume'].rolling(window=20).std()
            
            # Normalize features
            features_to_scale = [
                'open_price', 'high_price', 'low_price', 'close_price',
                'volume', 'quote_volume', 'rsi', 'macd', 'macd_signal'
            ]
            
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = RobustScaler()
                
            df[features_to_scale] = self.scalers[scaler_key].fit_transform(df[features_to_scale])
            
            return df.dropna()
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {str(e)}")
            raise

    def create_dataset(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        sequence_length: int,
        prediction_horizon: int,
        batch_size: int,
        train_split: float = 0.8
    ) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation datasets"""
        try:
            # Fetch and preprocess data
            df = asyncio.run(self.fetch_market_data(symbol, start_time, end_time))
            df = self.preprocess_data(df)
            
            # Save processed data
            data_file = self.data_path / f"{symbol}_{start_time.date()}_{end_time.date()}.parquet"
            df.to_parquet(data_file)
            
            # Split data
            train_size = int(len(df) * train_split)
            train_df = df.iloc[:train_size]
            val_df = df.iloc[train_size:]
            
            # Create datasets
            feature_columns = [
                'open_price', 'high_price', 'low_price', 'close_price',
                'volume', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
                'momentum', 'volume_ma', 'volume_std'
            ]
            
            train_dataset = CryptoDataset(
                data_file,
                sequence_length,
                prediction_horizon,
                feature_columns,
                'close_price',
                transform=None,
                train_mode=True
            )
            
            val_dataset = CryptoDataset(
                data_file,
                sequence_length,
                prediction_horizon,
                feature_columns,
                'close_price',
                transform=None,
                train_mode=True
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )
            
            return train_loader, val_loader
            
        except Exception as e:
            self.logger.error(f"Dataset creation failed: {str(e)}")
            raise

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calculate_macd(
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line"""
        exp1 = prices.ewm(span=fast_period).mean()
        exp2 = prices.ewm(span=slow_period).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period).mean()
        return macd, signal

    @staticmethod
    def _calculate_bollinger_bands(
        prices: pd.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        return upper, lower