from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
import logging
from datetime import datetime, timedelta
from enum import Enum

class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class RecommendationEngine:
    def __init__(self, config: Dict):
        """Initialize recommendation engine"""
        self.config = config
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.logger = logging.getLogger(__name__)
        
    async def generate_recommendation(self, token_data: Dict) -> Dict:
        """Generate trading recommendation based on analysis"""
        try:
            # Generate different types of analysis
            technical_signals = await self._analyze_technical_indicators(token_data)
            liquidity_analysis = self._analyze_liquidity(token_data)
            sentiment_score = await self._analyze_market_sentiment(token_data)
            
            # Combine analyses for final recommendation
            recommendation = self._combine_signals(
                technical_signals,
                liquidity_analysis,
                sentiment_score
            )
            
            return {
                "status": "success",
                "recommendation": recommendation,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendation: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def _analyze_technical_indicators(self, data: Dict) -> Dict:
        """Analyze technical indicators"""
        try:
            price_data = data.get("price_history", [])
            
            if not price_data:
                return {"signal": SignalType.NEUTRAL, "confidence": 0.0}
            
            # Calculate technical indicators
            rsi = self._calculate_rsi(price_data)
            macd = self._calculate_macd(price_data)
            moving_averages = self._calculate_moving_averages(price_data)
            
            # Determine signal based on indicators
            signal = self._evaluate_technical_signals(
                rsi=rsi,
                macd=macd,
                moving_averages=moving_averages
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Technical analysis failed: {e}")
            return {"signal": SignalType.NEUTRAL, "confidence": 0.0}

    def _analyze_liquidity(self, data: Dict) -> Dict:
        """Analyze token liquidity"""
        try:
            liquidity = data.get("liquidity", {})
            
            # Check liquidity metrics
            total_liquidity = liquidity.get("total_value", 0)
            depth_ratio = liquidity.get("depth_ratio", 0)
            volume_24h = liquidity.get("volume_24h", 0)
            
            # Calculate liquidity score
            score = self._calculate_liquidity_score(
                total_liquidity,
                depth_ratio,
                volume_24h
            )
            
            return {
                "score": score,
                "is_sufficient": score >= self.config.get("min_liquidity_score", 0.6)
            }
            
        except Exception as e:
            self.logger.error(f"Liquidity analysis failed: {e}")
            return {"score": 0, "is_sufficient": False}

    async def _analyze_market_sentiment(self, data: Dict) -> float:
        """Analyze market sentiment using AI"""
        try:
            # Prepare sentiment data
            sentiment_data = {
                "price_change": data.get("price_change_24h", 0),
                "volume_change": data.get("volume_change_24h", 0),
                "social_mentions": data.get("social_data", {}),
                "holder_changes": data.get("holder_changes", {})
            }
            
            # Generate sentiment analysis using LLM
            prompt = self._create_sentiment_prompt(sentiment_data)
            response = await self.llm.agenerate([prompt])
            
            # Parse sentiment score
            sentiment_score = self._parse_sentiment_score(response.text)
            return sentiment_score
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return 0.5  # Neutral sentiment as fallback

    def _combine_signals(
        self,
        technical: Dict,
        liquidity: Dict,
        sentiment: float
    ) -> Dict:
        """Combine different signals into final recommendation"""
        try:
            # Weight factors (configurable)
            weights = self.config.get("signal_weights", {
                "technical": 0.5,
                "liquidity": 0.3,
                "sentiment": 0.2
            })
            
            # Convert signals to numerical scores
            technical_score = self._signal_to_score(technical["signal"])
            
            # Calculate weighted average
            final_score = (
                technical_score * weights["technical"] +
                liquidity["score"] * weights["liquidity"] +
                sentiment * weights["sentiment"]
            )
            
            # Generate final recommendation
            return {
                "signal": self._score_to_signal(final_score),
                "confidence": min(technical["confidence"] * 1.2, 1.0),
                "factors": {
                    "technical": technical,
                    "liquidity": liquidity,
                    "sentiment": sentiment
                },
                "details": self._generate_recommendation_details(
                    final_score,
                    technical,
                    liquidity,
                    sentiment
                )
            }
            
        except Exception as e:
            self.logger.error(f"Signal combination failed: {e}")
            return {
                "signal": SignalType.NEUTRAL,
                "confidence": 0.0,
                "factors": {},
                "details": "Error generating recommendation"
            }

    def _calculate_rsi(self, price_data: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        try:
            if len(price_data) < period + 1:
                return 50  # Neutral RSI as fallback
                
            # Calculate price changes
            changes = [price_data[i+1] - price_data[i] for i in range(len(price_data)-1)]
            
            # Calculate gains and losses
            gains = [change if change > 0 else 0 for change in changes]
            losses = [-change if change < 0 else 0 for change in changes]
            
            # Calculate average gains and losses
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100
                
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"RSI calculation failed: {e}")
            return 50

    def _calculate_macd(
        self,
        price_data: List[float],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict:
        """Calculate MACD indicators"""
        try:
            if len(price_data) < slow_period + signal_period:
                return {"histogram": 0, "signal": 0}
            
            # Calculate EMAs
            fast_ema = self._calculate_ema(price_data, fast_period)
            slow_ema = self._calculate_ema(price_data, slow_period)
            
            # Calculate MACD line
            macd_line = fast_ema - slow_ema
            
            # Calculate signal line
            signal_line = self._calculate_ema([macd_line], signal_period)
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return {
                "histogram": histogram,
                "signal": signal_line
            }
            
        except Exception as e:
            self.logger.error(f"MACD calculation failed: {e}")
            return {"histogram": 0, "signal": 0}

    def _score_to_signal(self, score: float) -> SignalType:
        """Convert numerical score to signal type"""
        if score >= 0.8:
            return SignalType.STRONG_BUY
        elif score >= 0.6:
            return SignalType.BUY
        elif score <= 0.2:
            return SignalType.STRONG_SELL
        elif score <= 0.4:
            return SignalType.SELL
        else:
            return SignalType.NEUTRAL

    def _generate_recommendation_details(
        self,
        score: float,
        technical: Dict,
        liquidity: Dict,
        sentiment: float
    ) -> str:
        """Generate detailed recommendation explanation"""
        signal = self._score_to_signal(score)
        
        details = [
            f"Overall Signal: {signal.value}",
            f"Confidence Score: {score:.2f}",
            "",
            "Analysis Factors:"
        ]
        
        # Add technical analysis details
        details.extend([
            "Technical Analysis:",
            f"- Signal: {technical['signal'].value}",
            f"- Confidence: {technical['confidence']:.2f}"
        ])
        
        # Add liquidity analysis details
        details.extend([
            "Liquidity Analysis:",
            f"- Score: {liquidity['score']:.2f}",
            f"- Status: {'Sufficient' if liquidity['is_sufficient'] else 'Insufficient'}"
        ])
        
        # Add sentiment details
        details.extend([
            "Market Sentiment:",
            f"- Score: {sentiment:.2f}",
            f"- Interpretation: {self._interpret_sentiment(sentiment)}"
        ])
        
        return "\n".join(details)

    def _interpret_sentiment(self, sentiment: float) -> str:
        """Interpret sentiment score"""
        if sentiment >= 0.7:
            return "Very Positive"
        elif sentiment >= 0.6:
            return "Positive"
        elif sentiment <= 0.3:
            return "Very Negative"
        elif sentiment <= 0.4:
            return "Negative"
        else:
            return "Neutral"