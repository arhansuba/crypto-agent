from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import asyncio
import numpy as np
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.agent_toolkits import CdpToolkit

class MarketAnalyzer:
    """
    Advanced market analysis system that leverages CDP toolkit for comprehensive
    market analysis and trend detection.
    """
    
    def __init__(self, config: Dict):
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        
        # Analysis parameters
        self.volatility_window = config.get('volatility_window', 24)  # hours
        self.trend_periods = config.get('trend_periods', [1, 4, 24])  # hours
        self.volume_threshold = Decimal(str(config.get('volume_threshold', '10000')))
        
        # Cache management
        self.price_cache: Dict[str, List[Tuple[datetime, Decimal]]] = {}
        self.market_metrics: Dict[str, Dict] = {}
        self.trend_data: Dict[str, Dict] = {}

    async def analyze_market(self, token_address: str) -> Dict:
        """
        Perform comprehensive market analysis for a token.
        
        Args:
            token_address: Token address to analyze
            
        Returns:
            Comprehensive market analysis
        """
        tools = self.toolkit.get_tools()
        
        # Get current market data
        price_data = await self._get_price_data(token_address)
        volume_data = await tools.get_volume_data(token_address)
        
        # Calculate key metrics
        volatility = self._calculate_volatility(price_data)
        trends = self._analyze_trends(price_data)
        support_resistance = await self._find_support_resistance(price_data)
        
        # Calculate market metrics
        metrics = {
            'price_metrics': await self._analyze_price_metrics(price_data),
            'volume_metrics': self._analyze_volume_metrics(volume_data),
            'volatility': volatility,
            'trends': trends,
            'support_resistance': support_resistance,
            'momentum_indicators': await self._calculate_momentum_indicators(price_data),
            'timestamp': datetime.utcnow()
        }
        
        # Update cache
        self.market_metrics[token_address] = metrics
        
        return metrics

    async def detect_patterns(self, token_address: str) -> Dict:
        """
        Detect technical patterns in market data.
        
        Args:
            token_address: Token to analyze
            
        Returns:
            Detected patterns and confidence levels
        """
        price_data = await self._get_price_data(token_address)
        
        patterns = {
            'trend_patterns': self._detect_trend_patterns(price_data),
            'reversal_patterns': await self._detect_reversal_patterns(price_data),
            'continuation_patterns': self._detect_continuation_patterns(price_data)
        }
        
        return {
            'patterns': patterns,
            'confidence_scores': self._calculate_pattern_confidence(patterns),
            'timestamp': datetime.utcnow()
        }

    async def analyze_liquidity(self, token_address: str) -> Dict:
        """
        Analyze market liquidity metrics.
        
        Args:
            token_address: Token to analyze
            
        Returns:
            Liquidity analysis results
        """
        tools = self.toolkit.get_tools()
        
        pools = await tools.get_token_pools(token_address)
        total_liquidity = Decimal('0')
        
        pool_metrics = {}
        for pool in pools:
            metrics = await self._analyze_pool_liquidity(pool)
            pool_metrics[pool['address']] = metrics
            total_liquidity += metrics['total_liquidity']

        return {
            'total_liquidity': total_liquidity,
            'pool_metrics': pool_metrics,
            'depth_analysis': await self._analyze_market_depth(token_address),
            'liquidity_score': self._calculate_liquidity_score(total_liquidity, pool_metrics),
            'timestamp': datetime.utcnow()
        }

    async def _get_price_data(self, token_address: str) -> List[Tuple[datetime, Decimal]]:
        """Get historical price data with caching."""
        tools = self.toolkit.get_tools()
        
        if token_address not in self.price_cache:
            # Get historical prices
            prices = await tools.get_historical_prices(
                token_address,
                hours=self.volatility_window
            )
            
            self.price_cache[token_address] = prices
            
        return self.price_cache[token_address]

    def _calculate_volatility(self, price_data: List[Tuple[datetime, Decimal]]) -> Dict:
        """Calculate volatility metrics."""
        if not price_data:
            return {'hourly': Decimal('0'), 'daily': Decimal('0')}
            
        prices = [price for _, price in price_data]
        returns = [
            (prices[i] - prices[i-1]) / prices[i-1]
            for i in range(1, len(prices))
        ]
        
        hourly_vol = Decimal(str(np.std(returns) * np.sqrt(24)))
        daily_vol = Decimal(str(np.std(returns) * np.sqrt(365)))
        
        return {
            'hourly': hourly_vol,
            'daily': daily_vol,
            'annualized': daily_vol * Decimal('365').sqrt()
        }

    async def _analyze_price_metrics(self, price_data: List[Tuple[datetime, Decimal]]) -> Dict:
        """Analyze price movement metrics."""
        if not price_data:
            return {}
            
        current_price = price_data[-1][1]
        price_changes = {}
        
        for period in self.trend_periods:
            period_start = datetime.utcnow() - timedelta(hours=period)
            period_prices = [
                price for timestamp, price in price_data
                if timestamp >= period_start
            ]
            
            if period_prices:
                change = (current_price - period_prices[0]) / period_prices[0]
                price_changes[f'{period}h'] = change
                
        return {
            'current_price': current_price,
            'price_changes': price_changes,
            'moving_averages': self._calculate_moving_averages(price_data),
            'price_momentum': self._calculate_price_momentum(price_data)
        }

    def _log_error(self, message: str, error: Exception) -> None:
        """Log error with details."""
        error_details = {
            'message': message,
            'error': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }
        print(f"Market Analyzer Error: {error_details}")