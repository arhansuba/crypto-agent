"""
Advanced sentiment analysis system for cryptocurrency market analysis.
Combines social media sentiment, news analysis, and market indicators.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import re
from dataclasses import dataclass
import asyncio
import aiohttp
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

@dataclass
class SentimentData:
    """Represents analyzed sentiment data"""
    timestamp: datetime
    social_sentiment: float
    news_sentiment: float
    market_sentiment: float
    combined_score: float
    confidence: float
    source_metrics: Dict[str, float]
    keyword_sentiments: Dict[str, float]

class SentimentAnalyzer:
    """
    Comprehensive sentiment analysis system combining multiple data sources
    and analysis methods for market sentiment evaluation.
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """Initialize the sentiment analysis system"""
        self.config = config
        self.logger = logger
        
        # Initialize sentiment analyzers
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize neural network for sentiment fusion
        self.sentiment_fusion_model = self._initialize_fusion_model()
        
        # Initialize scalers
        self.sentiment_scaler = StandardScaler()
        
        # Load sentiment lexicons
        self.crypto_lexicon = self._load_crypto_lexicon()
        
        # Initialize state tracking
        self.sentiment_history = []
        self.source_weights = self._calculate_source_weights()
        
        self.logger.info("Sentiment analysis system initialized")

    def _initialize_fusion_model(self) -> nn.Module:
        """Initialize neural network for sentiment fusion"""
        class SentimentFusionNet(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Tanh()
                )
                
                self.confidence = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                features = self.network[0](x)
                sentiment = self.network(x)
                confidence = self.confidence(features)
                return sentiment, confidence
        
        model = SentimentFusionNet(
            input_dim=self.config['sentiment']['fusion_input_dim'],
            hidden_dim=self.config['sentiment']['fusion_hidden_dim']
        )
        
        model.load_state_dict(torch.load(
            self.config['sentiment']['fusion_model_path']
        ))
        
        return model.eval()

    def _load_crypto_lexicon(self) -> Dict[str, float]:
        """Load cryptocurrency-specific sentiment lexicon"""
        try:
            with open(self.config['sentiment']['lexicon_path'], 'r') as f:
                return {
                    word: float(score)
                    for word, score in [line.strip().split('\t')
                    for line in f if line.strip()]
                }
        except Exception as e:
            self.logger.error(f"Failed to load crypto lexicon: {str(e)}")
            return {}

    def _calculate_source_weights(self) -> Dict[str, float]:
        """Calculate weights for different sentiment sources"""
        weights = {
            'social': 0.4,
            'news': 0.3,
            'market': 0.3
        }
        
        if self.config['sentiment'].get('source_weights'):
            weights.update(self.config['sentiment']['source_weights'])
        
        return weights

    async def analyze_text_sentiment(
        self,
        text: str,
        source_type: str
    ) -> Dict[str, float]:
        """Analyze sentiment of text with context awareness"""
        try:
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Get VADER sentiment
            vader_scores = self.vader.polarity_scores(cleaned_text)
            
            # Get TextBlob sentiment
            blob = TextBlob(cleaned_text)
            textblob_sentiment = blob.sentiment.polarity
            
            # Apply crypto-specific lexicon
            crypto_sentiment = self._apply_crypto_lexicon(cleaned_text)
            
            # Combine sentiment scores with source-specific weights
            if source_type == 'news':
                weights = (0.4, 0.3, 0.3)  # VADER, TextBlob, Crypto
            else:
                weights = (0.3, 0.3, 0.4)  # More weight on crypto for social
            
            combined_sentiment = (
                weights[0] * vader_scores['compound'] +
                weights[1] * textblob_sentiment +
                weights[2] * crypto_sentiment
            )
            
            return {
                'sentiment_score': combined_sentiment,
                'vader_score': vader_scores['compound'],
                'textblob_score': textblob_sentiment,
                'crypto_score': crypto_sentiment,
                'subjectivity': blob.sentiment.subjectivity,
                'keyword_sentiments': self._extract_keyword_sentiments(cleaned_text)
            }
            
        except Exception as e:
            self.logger.error(f"Text sentiment analysis failed: {str(e)}")
            raise

    async def analyze_market_sentiment(
        self,
        price_data: np.ndarray,
        volume_data: np.ndarray
    ) -> Dict[str, float]:
        """Analyze market sentiment from price and volume data"""
        try:
            # Calculate technical indicators
            rsi = self._calculate_rsi(price_data)
            macd = self._calculate_macd(price_data)
            volume_profile = self._analyze_volume_profile(volume_data)
            
            # Calculate market sentiment score
            market_sentiment = np.mean([
                (rsi - 50) / 50,  # Normalize RSI to [-1, 1]
                np.sign(macd) * min(abs(macd), 1),  # Normalized MACD
                volume_profile['sentiment']
            ])
            
            return {
                'market_sentiment': float(market_sentiment),
                'rsi_sentiment': float((rsi - 50) / 50),
                'macd_sentiment': float(np.sign(macd) * min(abs(macd), 1)),
                'volume_sentiment': float(volume_profile['sentiment']),
                'trend_strength': float(volume_profile['trend_strength'])
            }
            
        except Exception as e:
            self.logger.error(f"Market sentiment analysis failed: {str(e)}")
            raise

    @torch.no_grad()
    async def combine_sentiment_sources(
        self,
        social_sentiment: Dict[str, float],
        news_sentiment: Dict[str, float],
        market_sentiment: Dict[str, float]
    ) -> SentimentData:
        """Combine sentiment from multiple sources using neural network"""
        try:
            # Prepare input features
            features = np.array([
                social_sentiment['sentiment_score'],
                social_sentiment['subjectivity'],
                news_sentiment['sentiment_score'],
                news_sentiment['subjectivity'],
                market_sentiment['market_sentiment'],
                market_sentiment['trend_strength']
            ])
            
            # Scale features
            scaled_features = self.sentiment_scaler.fit_transform(
                features.reshape(1, -1)
            )
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(scaled_features)
            
            # Get model prediction
            sentiment_score, confidence = self.sentiment_fusion_model(input_tensor)
            
            # Create sentiment data object
            sentiment_data = SentimentData(
                timestamp=datetime.now(),
                social_sentiment=social_sentiment['sentiment_score'],
                news_sentiment=news_sentiment['sentiment_score'],
                market_sentiment=market_sentiment['market_sentiment'],
                combined_score=float(sentiment_score),
                confidence=float(confidence),
                source_metrics={
                    'social_subjectivity': social_sentiment['subjectivity'],
                    'news_subjectivity': news_sentiment['subjectivity'],
                    'market_trend_strength': market_sentiment['trend_strength']
                },
                keyword_sentiments=self._combine_keyword_sentiments(
                    social_sentiment.get('keyword_sentiments', {}),
                    news_sentiment.get('keyword_sentiments', {})
                )
            )
            
            # Update sentiment history
            self.sentiment_history.append(sentiment_data)
            if len(self.sentiment_history) > self.config['sentiment']['history_size']:
                self.sentiment_history.pop(0)
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Sentiment combination failed: {str(e)}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?]', '', text)
        
        return text.strip()

    def _apply_crypto_lexicon(self, text: str) -> float:
        """Apply cryptocurrency-specific sentiment lexicon"""
        words = text.split()
        sentiment_scores = []
        
        for word in words:
            if word in self.crypto_lexicon:
                sentiment_scores.append(self.crypto_lexicon[word])
        
        return np.mean(sentiment_scores) if sentiment_scores else 0.0

    def _extract_keyword_sentiments(self, text: str) -> Dict[str, float]:
        """Extract sentiment scores for key cryptocurrency terms"""
        keywords = self.config['sentiment']['keywords']
        keyword_sentiments = {}
        
        for keyword in keywords:
            if keyword in text.lower():
                # Extract context around keyword
                context = self._extract_context(text, keyword)
                # Calculate sentiment for context
                sentiment = self._calculate_context_sentiment(context)
                keyword_sentiments[keyword] = sentiment
        
        return keyword_sentiments

    def _extract_context(self, text: str, keyword: str, window: int = 10) -> str:
        """Extract context window around keyword"""
        words = text.split()
        try:
            idx = words.index(keyword)
            start = max(0, idx - window)
            end = min(len(words), idx + window + 1)
            return ' '.join(words[start:end])
        except ValueError:
            return ''

    def _calculate_context_sentiment(self, context: str) -> float:
        """Calculate sentiment score for a context window"""
        if not context:
            return 0.0
        
        vader_score = self.vader.polarity_scores(context)['compound']
        textblob_score = TextBlob(context).sentiment.polarity
        crypto_score = self._apply_crypto_lexicon(context)
        
        return np.mean([vader_score, textblob_score, crypto_score])

    @staticmethod
    def _calculate_rsi(
        prices: np.ndarray,
        period: int = 14
    ) -> float:
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].mean()
        down = -seed[seed < 0].mean()
        
        if down != 0:
            rs = up/down
        else:
            rs = float('inf')
        
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _calculate_macd(
        prices: np.ndarray,
        fast_period: int = 12,
        slow_period: int = 26
    ) -> float:
        """Calculate MACD value"""
        exp1 = pd.Series(prices).ewm(span=fast_period).mean()
        exp2 = pd.Series(prices).ewm(span=slow_period).mean()
        macd = exp1 - exp2
        return macd.iloc[-1]

    def _analyze_volume_profile(
        self,
        volume_data: np.ndarray
    ) -> Dict[str, float]:
        """Analyze volume profile for sentiment indicators"""
        # Calculate volume metrics
        volume_ma = np.mean(volume_data[-20:])
        volume_std = np.std(volume_data[-20:])
        
        # Calculate trend strength
        trend_strength = volume_ma / (volume_std + 1e-6)
        
        # Calculate volume sentiment
        recent_volume = np.mean(volume_data[-5:])
        volume_change = (recent_volume - volume_ma) / volume_ma
        volume_sentiment = np.tanh(volume_change)  # Normalize to [-1, 1]
        
        return {
            'sentiment': float(volume_sentiment),
            'trend_strength': float(min(trend_strength, 1.0))
        }

    def _combine_keyword_sentiments(
        self,
        social_keywords: Dict[str, float],
        news_keywords: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine keyword sentiments from different sources"""
        combined = {}
        
        # Combine sentiments for overlapping keywords
        all_keywords = set(social_keywords) | set(news_keywords)
        
        for keyword in all_keywords:
            social_score = social_keywords.get(keyword, 0)
            news_score = news_keywords.get(keyword, 0)
            
            if keyword in social_keywords and keyword in news_keywords:
                combined[keyword] = 0.6 * social_score + 0.4 * news_score
            elif keyword in social_keywords:
                combined[keyword] = social_score
            else:
                combined[keyword] = news_score
        
        return combined