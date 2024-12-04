from typing import Dict, List, Optional, Tuple, Union
from decimal import Decimal
from datetime import datetime, timedelta
import asyncio
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.agent_toolkits import CdpToolkit
import numpy as np

class PositionCalculator:
    """Calculates optimal position sizes based on risk parameters and market conditions."""
    
    def __init__(self, config: Dict):
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        
        # Risk parameters
        self.max_position_size = Decimal(str(config.get('max_position_size', '5.0')))
        self.max_portfolio_risk = Decimal(str(config.get('max_portfolio_risk', '0.02')))  # 2%
        self.position_risk_limit = Decimal(str(config.get('position_risk_limit', '0.01')))  # 1%
        self.vol_lookback_days = config.get('vol_lookback_days', 30)

    async def calculate_position_size(
        self,
        token_address: str,
        portfolio_value: Decimal,
        risk_level: str = 'medium'
    ) -> Dict:
        """
        Calculate optimal position size based on risk metrics and portfolio value.
        
        Args:
            token_address: Token address to analyze
            portfolio_value: Current portfolio value
            risk_level: Risk level (low, medium, high)
            
        Returns:
            Position size calculation with risk metrics
        """
        # Get volatility metrics
        volatility = await self._calculate_volatility(token_address)
        
        # Get liquidity metrics
        liquidity = await self._analyze_liquidity(token_address)
        
        # Calculate risk-adjusted position size
        risk_multiplier = self._get_risk_multiplier(risk_level)
        max_risk_amount = portfolio_value * self.position_risk_limit * risk_multiplier
        
        position_size = self._calculate_risk_adjusted_size(
            max_risk_amount,
            volatility,
            liquidity
        )
        
        return {
            'recommended_size': position_size,
            'max_size': min(self.max_position_size, position_size * Decimal('1.5')),
            'risk_metrics': {
                'volatility': volatility,
                'liquidity_score': liquidity['score'],
                'risk_score': self._calculate_risk_score(volatility, liquidity)
            },
            'constraints': {
                'portfolio_limit': max_risk_amount,
                'liquidity_limit': liquidity['max_position'],
                'volatility_limit': max_risk_amount / volatility
            }
        }

    async def _calculate_volatility(self, token_address: str) -> Decimal:
        """Calculate token volatility over specified lookback period."""
        tools = self.toolkit.get_tools()
        
        # Get historical prices
        prices = await tools.get_historical_prices(
            token_address,
            days=self.vol_lookback_days
        )
        
        if not prices:
            return Decimal('1')  # Conservative default
            
        # Calculate daily returns
        returns = [
            (prices[i] - prices[i-1]) / prices[i-1]
            for i in range(1, len(prices))
        ]
        
        # Calculate annualized volatility
        daily_vol = Decimal(str(np.std(returns)))
        annual_vol = daily_vol * Decimal(str(np.sqrt(365)))
        
        return annual_vol

class PortfolioManager:
    """Manages portfolio risk and position allocation."""
    
    def __init__(self, config: Dict):
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        self.position_calculator = PositionCalculator(config)
        
        # Portfolio parameters
        self.max_position_count = config.get('max_position_count', 10)
        self.rebalance_threshold = Decimal(str(config.get('rebalance_threshold', '0.1')))
        self.min_position_size = Decimal(str(config.get('min_position_size', '0.01')))
        
        # State tracking
        self.positions: Dict[str, Dict] = {}
        self.portfolio_stats: Dict[str, Decimal] = {}
        self.rebalance_history: List[Dict] = []

    async def analyze_portfolio(self) -> Dict:
        """
        Perform comprehensive portfolio analysis.
        
        Returns:
            Portfolio analysis including risk metrics and recommendations
        """
        if not self.positions:
            return {"status": "empty", "total_value": Decimal('0')}
            
        # Calculate current positions value
        position_values = await self._calculate_position_values()
        
        # Calculate portfolio metrics
        metrics = await self._calculate_portfolio_metrics(position_values)
        
        # Generate recommendations
        recommendations = await self._generate_portfolio_recommendations(
            position_values,
            metrics
        )
        
        return {
            'status': 'active',
            'total_value': sum(position_values.values()),
            'position_count': len(self.positions),
            'metrics': metrics,
            'risk_analysis': await self._analyze_portfolio_risk(position_values),
            'recommendations': recommendations,
            'timestamp': datetime.utcnow()
        }

    async def rebalance_portfolio(self, target_allocations: Optional[Dict] = None) -> Dict:
        """
        Rebalance portfolio positions according to target allocations.
        
        Args:
            target_allocations: Optional target position allocations
            
        Returns:
            Rebalancing actions and results
        """
        current_values = await self._calculate_position_values()
        total_value = sum(current_values.values())
        
        if not target_allocations:
            target_allocations = await self._calculate_optimal_allocations(
                current_values,
                total_value
            )
            
        rebalance_actions = []
        for token, target in target_allocations.items():
            current = current_values.get(token, Decimal('0'))
            target_value = total_value * target
            
            if abs(current - target_value) / target_value > self.rebalance_threshold:
                action = await self._execute_rebalance(
                    token,
                    current,
                    target_value
                )
                rebalance_actions.append(action)
                
        return {
            'actions': rebalance_actions,
            'timestamp': datetime.utcnow(),
            'portfolio_value': total_value,
            'new_allocations': target_allocations
        }

    def _calculate_risk_score(self, volatility: Decimal, liquidity: Dict) -> Decimal:
        """Calculate comprehensive risk score."""
        vol_score = volatility / Decimal('2')  # Normalize volatility
        liq_score = Decimal('1') - liquidity['score']  # Invert liquidity score
        
        return (vol_score * Decimal('0.7') + liq_score * Decimal('0.3')).quantize(
            Decimal('0.01')
        )