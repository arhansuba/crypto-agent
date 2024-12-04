from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.agent_toolkits import CdpToolkit

class PatternDetector:
    """
    Advanced pattern detection system that identifies technical patterns
    and market formations using CDP's toolkit.
    """
    
    def __init__(self, config: Dict):
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        
        # Pattern parameters
        self.min_pattern_confidence = Decimal(str(config.get('min_pattern_confidence', '0.7')))
        self.lookback_periods = config.get('lookback_periods', [24, 72, 168])  # hours
        self.price_tolerance = Decimal(str(config.get('price_tolerance', '0.02')))
        
        # State tracking
        self.detected_patterns: Dict[str, List[Dict]] = {}
        self.formation_history: Dict[str, List[Dict]] = {}
        self.validation_metrics: Dict[str, Dict] = {}

    async def detect_patterns(self, token_address: str) -> Dict:
        """
        Detect and analyze technical patterns for a token.
        
        Args:
            token_address: Token to analyze
            
        Returns:
            Detected patterns with confidence scores
        """
        tools = self.toolkit.get_tools()
        
        # Get price data
        price_data = await tools.get_historical_prices(token_address)
        volume_data = await tools.get_volume_data(token_address)
        
        # Detect various pattern types
        trend_patterns = await self._detect_trend_patterns(price_data)
        reversal_patterns = await self._detect_reversal_patterns(price_data)
        continuation_patterns = await self._detect_continuation_patterns(price_data)
        volume_patterns = await self._analyze_volume_patterns(volume_data)
        
        # Validate and combine patterns
        combined_patterns = self._combine_pattern_signals(
            trend_patterns,
            reversal_patterns,
            continuation_patterns,
            volume_patterns
        )
        
        return {
            'patterns': combined_patterns,
            'confidence_metrics': self._calculate_confidence_metrics(combined_patterns),
            'validations': await self._validate_patterns(combined_patterns),
            'timestamp': datetime.utcnow()
        }

    async def _detect_trend_patterns(self, price_data: List[Tuple[datetime, Decimal]]) -> Dict:
        """Detect trend-based patterns."""
        patterns = {}
        
        # Uptrend detection
        uptrend = self._detect_uptrend(price_data)
        if uptrend['confidence'] > self.min_pattern_confidence:
            patterns['uptrend'] = uptrend
            
        # Downtrend detection
        downtrend = self._detect_downtrend(price_data)
        if downtrend['confidence'] > self.min_pattern_confidence:
            patterns['downtrend'] = downtrend
            
        # Channel detection
        channel = await self._detect_channel(price_data)
        if channel['confidence'] > self.min_pattern_confidence:
            patterns['channel'] = channel
            
        return patterns

    async def _detect_reversal_patterns(self, price_data: List[Tuple[datetime, Decimal]]) -> Dict:
        """Detect potential reversal patterns."""
        patterns = {}
        
        # Head and shoulders
        hand_s = self._detect_head_and_shoulders(price_data)
        if hand_s['confidence'] > self.min_pattern_confidence:
            patterns['head_and_shoulders'] = hand_s
            
        # Double top/bottom
        double_patterns = self._detect_double_patterns(price_data)
        patterns.update(double_patterns)
        
        # Triple top/bottom
        triple_patterns = self._detect_triple_patterns(price_data)
        patterns.update(triple_patterns)
        
        return patterns

    async def _detect_continuation_patterns(self, price_data: List[Tuple[datetime, Decimal]]) -> Dict:
        """Detect continuation patterns."""
        patterns = {}
        
        # Flag patterns
        flag = self._detect_flag_pattern(price_data)
        if flag['confidence'] > self.min_pattern_confidence:
            patterns['flag'] = flag
            
        # Triangle patterns
        triangles = await self._detect_triangles(price_data)
        patterns.update(triangles)
        
        # Rectangle patterns
        rectangle = self._detect_rectangle(price_data)
        if rectangle['confidence'] > self.min_pattern_confidence:
            patterns['rectangle'] = rectangle
            
        return patterns

    def _detect_uptrend(self, price_data: List[Tuple[datetime, Decimal]]) -> Dict:
        """Detect uptrend pattern."""
        if not price_data:
            return {'confidence': Decimal('0')}
            
        prices = [price for _, price in price_data]
        higher_lows = 0
        higher_highs = 0
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                higher_highs += 1
            if min(prices[i-1:i+1]) > min(prices[max(0, i-2):i]):
                higher_lows += 1
                
        confidence = Decimal(str((higher_lows + higher_highs) / (2 * (len(prices) - 1))))
        
        return {
            'type': 'uptrend',
            'confidence': confidence,
            'strength': self._calculate_trend_strength(prices),
            'support_levels': self._find_support_levels(prices)
        }

    def _detect_head_and_shoulders(self, price_data: List[Tuple[datetime, Decimal]]) -> Dict:
        """Detect head and shoulders pattern."""
        prices = [price for _, price in price_data]
        pattern_found = False
        confidence = Decimal('0')
        
        if len(prices) < 5:
            return {'confidence': confidence}
            
        for i in range(2, len(prices)-2):
            # Check left shoulder
            if prices[i-2] < prices[i-1] > prices[i]:
                # Check head
                if prices[i] < prices[i+1] > prices[i+2]:
                    # Check right shoulder
                    if prices[i+1] > prices[i+2] and prices[i-1] == prices[i+2]:
                        pattern_found = True
                        confidence = self._calculate_pattern_confidence(prices[i-2:i+3])
                        break
                        
        return {
            'type': 'head_and_shoulders',
            'confidence': confidence,
            'neckline': self._calculate_neckline(prices) if pattern_found else None,
            'target': self._calculate_pattern_target(prices) if pattern_found else None
        }

    def _log_error(self, message: str, error: Exception) -> None:
        """Log error with details."""
        error_details = {
            'message': message,
            'error': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }
        print(f"Pattern Detector Error: {error_details}")  # Replace with proper logging