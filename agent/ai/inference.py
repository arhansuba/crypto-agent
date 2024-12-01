"""
Inference system implementation for AI crypto trading models.
Provides real-time prediction capabilities with confidence estimation and performance monitoring.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import logging
import asyncio
from pathlib import Path
import json
from dataclasses import dataclass
import pandas as pd
from collections import deque

@dataclass
class InferencePrediction:
    """Represents a single model prediction with metadata"""
    timestamp: datetime
    price_prediction: float
    confidence_score: float
    trend_direction: str
    risk_metrics: Dict[str, float]
    support_levels: List[float]
    resistance_levels: List[float]
    metadata: Dict[str, Any]

class ModelInference:
    """
    Inference engine for market prediction and risk analysis models.
    Handles real-time predictions and monitoring.
    """
    
    def __init__(
        self,
        market_predictor: torch.nn.Module,
        risk_analyzer: torch.nn.Module,
        config: Dict,
        logger: logging.Logger
    ):
        self.market_predictor = market_predictor
        self.risk_analyzer = risk_analyzer
        self.config = config
        self.logger = logger
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.market_predictor.to(self.device)
        self.risk_analyzer.to(self.device)
        
        # Initialize prediction history
        self.prediction_history = deque(
            maxlen=config['inference']['history_size']
        )
        
        # Load preprocessing components
        self.scaler = self._load_scaler()
        
        # Initialize performance metrics
        self.performance_metrics = {
            'prediction_accuracy': [],
            'trend_accuracy': [],
            'confidence_correlation': []
        }
        
        self.logger.info(f"Inference engine initialized on device: {self.device}")

    def _load_scaler(self) -> Any:
        """Load data scaler for preprocessing"""
        try:
            import joblib
            scaler_path = Path(self.config['preprocessing']['scaler_path'])
            return joblib.load(scaler_path)
            
        except Exception as e:
            self.logger.error(f"Failed to load scaler: {str(e)}")
            raise

    async def preprocess_data(
        self,
        market_data: Dict[str, np.ndarray]
    ) -> torch.Tensor:
        """Preprocess input data for inference"""
        try:
            # Extract features
            features = self._extract_features(market_data)
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            
            # Convert to tensor
            tensor_features = torch.FloatTensor(scaled_features).to(self.device)
            
            return tensor_features
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {str(e)}")
            raise

    def _extract_features(self, market_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract and calculate features from market data"""
        try:
            price_data = market_data['prices']
            volume_data = market_data['volumes']
            
            # Calculate technical indicators
            features = []
            
            # Price-based features
            features.extend([
                self._calculate_returns(price_data),
                self._calculate_volatility(price_data),
                self._calculate_rsi(price_data),
                self._calculate_macd(price_data)
            ])
            
            # Volume-based features
            features.extend([
                volume_data,
                self._calculate_volume_ma(volume_data),
                self._calculate_volume_std(volume_data)
            ])
            
            return np.column_stack(features)
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            raise

    @staticmethod
    def _calculate_returns(prices: np.ndarray, period: int = 1) -> np.ndarray:
        """Calculate price returns"""
        return np.concatenate([[0] * period, np.diff(np.log(prices), period)])

    @staticmethod
    def _calculate_volatility(
        prices: np.ndarray,
        window: int = 20
    ) -> np.ndarray:
        """Calculate rolling volatility"""
        returns = np.diff(np.log(prices))
        volatility = np.array([np.std(returns[max(0, i-window):i])
                             for i in range(1, len(returns)+1)])
        return np.concatenate([[0], volatility])

    @staticmethod
    def _calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator"""
        returns = np.diff(prices)
        gains = np.maximum(returns, 0)
        losses = -np.minimum(returns, 0)
        
        avg_gain = np.concatenate([[0], gains]).cumsum()
        avg_loss = np.concatenate([[0], losses]).cumsum()
        
        rs = avg_gain / np.maximum(avg_loss, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    @staticmethod
    def _calculate_macd(
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26
    ) -> np.ndarray:
        """Calculate MACD indicator"""
        exp1 = pd.Series(prices).ewm(span=fast).mean()
        exp2 = pd.Series(prices).ewm(span=slow).mean()
        macd = exp1 - exp2
        return macd.values

    @staticmethod
    def _calculate_volume_ma(
        volumes: np.ndarray,
        window: int = 20
    ) -> np.ndarray:
        """Calculate volume moving average"""
        return pd.Series(volumes).rolling(window=window).mean().fillna(0).values

    @staticmethod
    def _calculate_volume_std(
        volumes: np.ndarray,
        window: int = 20
    ) -> np.ndarray:
        """Calculate volume standard deviation"""
        return pd.Series(volumes).rolling(window=window).std().fillna(0).values

    @torch.no_grad()
    async def predict(
        self,
        market_data: Dict[str, np.ndarray],
        portfolio_state: Optional[Dict] = None
    ) -> InferencePrediction:
        """Generate predictions using the models"""
        try:
            # Preprocess data
            features = await self.preprocess_data(market_data)
            
            # Generate market prediction
            self.market_predictor.eval()
            market_prediction = self.market_predictor(features.unsqueeze(0))
            
            # Generate risk assessment
            self.risk_analyzer.eval()
            risk_assessment = self.risk_analyzer(features.unsqueeze(0))
            
            # Process predictions
            price_pred = self.scaler.inverse_transform(
                market_prediction[0].cpu().numpy()
            )[0]
            
            confidence = float(market_prediction[1].cpu().numpy()[0])
            
            trend = "up" if price_pred > market_data['prices'][-1] else "down"
            
            # Calculate support/resistance levels
            support, resistance = self._calculate_price_levels(
                market_data['prices'],
                price_pred
            )
            
            # Create prediction object
            prediction = InferencePrediction(
                timestamp=datetime.now(),
                price_prediction=float(price_pred),
                confidence_score=confidence,
                trend_direction=trend,
                risk_metrics=self._process_risk_metrics(risk_assessment),
                support_levels=support,
                resistance_levels=resistance,
                metadata={
                    'market_conditions': self._analyze_market_conditions(market_data),
                    'portfolio_impact': self._analyze_portfolio_impact(
                        portfolio_state
                    ) if portfolio_state else None
                }
            )
            
            # Update prediction history
            self.prediction_history.append(prediction)
            
            # Update performance metrics if actual price available
            if len(self.prediction_history) > 1:
                self._update_performance_metrics(
                    self.prediction_history[-2],
                    market_data['prices'][-1]
                )
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {str(e)}")
            raise

    def _calculate_price_levels(
        self,
        price_history: np.ndarray,
        predicted_price: float,
        num_levels: int = 3
    ) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels"""
        try:
            # Find local minima and maxima
            prices = price_history[-100:]  # Use recent price history
            
            support_levels = []
            resistance_levels = []
            
            # Simple level detection using rolling windows
            for i in range(1, len(prices) - 1):
                if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                    support_levels.append(prices[i])
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                    resistance_levels.append(prices[i])
            
            # Sort and filter levels
            support_levels = sorted(support_levels)[-num_levels:]
            resistance_levels = sorted(resistance_levels)[:num_levels]
            
            return support_levels, resistance_levels
            
        except Exception as e:
            self.logger.error(f"Level calculation failed: {str(e)}")
            return [], []

    def _process_risk_metrics(self, risk_assessment: torch.Tensor) -> Dict[str, float]:
        """Process risk assessment outputs"""
        risk_components = risk_assessment.cpu().numpy()[0]
        
        return {
            'volatility_risk': float(risk_components[0]),
            'liquidity_risk': float(risk_components[1]),
            'market_risk': float(risk_components[2]),
            'overall_risk': float(np.mean(risk_components))
        }

    def _analyze_market_conditions(
        self,
        market_data: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze current market conditions"""
        try:
            recent_prices = market_data['prices'][-20:]
            recent_volumes = market_data['volumes'][-20:]
            
            return {
                'price_trend': self._calculate_trend_strength(recent_prices),
                'volume_profile': np.mean(recent_volumes) / np.std(recent_volumes),
                'market_volatility': np.std(self._calculate_returns(recent_prices))
            }
            
        except Exception as e:
            self.logger.error(f"Market analysis failed: {str(e)}")
            return {}

    @staticmethod
    def _calculate_trend_strength(prices: np.ndarray) -> float:
        """Calculate trend strength indicator"""
        returns = np.diff(np.log(prices))
        positive_returns = np.sum(returns > 0)
        return float(positive_returns / len(returns))

    def _analyze_portfolio_impact(
        self,
        portfolio_state: Dict[str, float]
    ) -> Dict[str, float]:
        """Analyze prediction impact on portfolio"""
        try:
            current_exposure = portfolio_state.get('position_size', 0)
            available_margin = portfolio_state.get('available_margin', 0)
            
            return {
                'max_position_size': min(
                    available_margin,
                    self.config['risk']['max_position_size']
                ),
                'exposure_ratio': current_exposure / available_margin
                if available_margin > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio analysis failed: {str(e)}")
            return {}

    def _update_performance_metrics(
        self,
        prediction: InferencePrediction,
        actual_price: float
    ):
        """Update prediction performance metrics"""
        try:
            # Calculate prediction accuracy
            accuracy = 1 - abs(
                prediction.price_prediction - actual_price
            ) / actual_price
            self.performance_metrics['prediction_accuracy'].append(accuracy)
            
            # Calculate trend accuracy
            predicted_trend = prediction.trend_direction
            actual_trend = "up" if actual_price > prediction.price_prediction else "down"
            self.performance_metrics['trend_accuracy'].append(
                predicted_trend == actual_trend
            )
            
            # Calculate confidence correlation
            confidence_correlation = prediction.confidence_score * accuracy
            self.performance_metrics['confidence_correlation'].append(
                confidence_correlation
            )
            
            # Trim metrics history if needed
            max_history = self.config['inference']['metrics_history_size']
            for metric_list in self.performance_metrics.values():
                if len(metric_list) > max_history:
                    metric_list.pop(0)
                    
        except Exception as e:
            self.logger.error(f"Metrics update failed: {str(e)}")

    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of prediction performance"""
        try:
            return {
                'avg_prediction_accuracy': np.mean(
                    self.performance_metrics['prediction_accuracy']
                ),
                'avg_trend_accuracy': np.mean(
                    self.performance_metrics['trend_accuracy']
                ),
                'confidence_correlation': np.mean(
                    self.performance_metrics['confidence_correlation']
                )
            }
            
        except Exception as e:
            self.logger.error(f"Performance summary failed: {str(e)}")
            return {}