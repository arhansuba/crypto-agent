from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper

class MarketAnalyzer:
    def __init__(self, config: Dict):
        """Initialize market analyzer with CDP integration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        self.tools = self.toolkit.get_tools()

    async def analyze_token(self, token_address: str, analysis_depth: str = "basic") -> Dict:
        """Analyze token with specified depth using CDP tools"""
        try:
            analysis = {
                "timestamp": datetime.utcnow().isoformat(),
                "token_address": token_address
            }

            # Basic analysis for all tiers
            basic_info = await self.tools.get_token_info(token_address)
            analysis["basic"] = {
                "price": basic_info.get("price"),
                "volume_24h": basic_info.get("volume"),
                "market_cap": basic_info.get("market_cap"),
                "total_supply": basic_info.get("total_supply")
            }

            if analysis_depth in ["advanced", "premium"]:
                # Add liquidity analysis
                liquidity = await self.tools.analyze_liquidity(token_address)
                analysis["liquidity"] = {
                    "total_liquidity": liquidity.get("total_liquidity"),
                    "liquidity_pairs": liquidity.get("pairs", []),
                    "depth_score": liquidity.get("depth_score")
                }

                # Add holder analysis
                holder_info = await self.tools.get_token_holders(token_address)
                analysis["holder_metrics"] = self._analyze_holder_distribution(holder_info)

            if analysis_depth == "premium":
                # Add advanced metrics and patterns
                analysis["technical"] = await self._analyze_technical_metrics(token_address)
                analysis["patterns"] = await self._detect_patterns(token_address)

            return {"status": "success", "analysis": analysis}

        except Exception as e:
            self.logger.error(f"Token analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    async def analyze_liquidity_pool(self, pool_address: str) -> Dict:
        """Analyze liquidity pool metrics"""
        try:
            pool_info = await self.tools.get_pool_info(pool_address)
            
            return {
                "status": "success",
                "pool_data": {
                    "total_liquidity": pool_info.get("total_liquidity"),
                    "token0_reserve": pool_info.get("token0_reserve"),
                    "token1_reserve": pool_info.get("token1_reserve"),
                    "fee_tier": pool_info.get("fee"),
                    "volume_24h": pool_info.get("volume_24h"),
                    "price_impact": await self._calculate_price_impact(pool_info)
                }
            }
        except Exception as e:
            self.logger.error(f"Pool analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    async def get_market_metrics(self) -> Dict:
        """Get overall market metrics"""
        try:
            market_data = await self.tools.get_market_metrics()
            
            return {
                "status": "success",
                "metrics": {
                    "total_volume_24h": market_data.get("total_volume"),
                    "total_liquidity": market_data.get("total_liquidity"),
                    "active_pairs": market_data.get("active_pairs"),
                    "gas_price": market_data.get("gas_price")
                }
            }
        except Exception as e:
            self.logger.error(f"Market metrics failed: {e}")
            return {"status": "error", "message": str(e)}

    def _analyze_holder_distribution(self, holder_info: Dict) -> Dict:
        """Analyze token holder distribution"""
        total_holders = len(holder_info.get("holders", []))
        holdings = sorted(holder_info.get("holders", []), key=lambda x: x.get("balance", 0), reverse=True)

        # Calculate concentration metrics
        top_10_holdings = sum(h.get("balance", 0) for h in holdings[:10])
        total_supply = sum(h.get("balance", 0) for h in holdings)

        return {
            "total_holders": total_holders,
            "top_10_concentration": (top_10_holdings / total_supply) * 100 if total_supply else 0,
            "distribution_score": self._calculate_distribution_score(holdings, total_supply),
            "holder_breakdown": {
                "large_holders": len([h for h in holdings if h.get("balance", 0) / total_supply > 0.01]),
                "medium_holders": len([h for h in holdings if 0.001 < h.get("balance", 0) / total_supply <= 0.01]),
                "small_holders": len([h for h in holdings if h.get("balance", 0) / total_supply <= 0.001])
            }
        }

    async def _analyze_technical_metrics(self, token_address: str) -> Dict:
        """Calculate technical analysis metrics"""
        try:
            price_history = await self.tools.get_price_history(token_address)
            
            return {
                "price_change_24h": self._calculate_price_change(price_history, hours=24),
                "volume_change_24h": self._calculate_volume_change(price_history, hours=24),
                "volatility": self._calculate_volatility(price_history),
                "momentum_indicators": await self._calculate_momentum_indicators(price_history)
            }
        except Exception as e:
            self.logger.error(f"Technical analysis failed: {e}")
            return {}

    async def _detect_patterns(self, token_address: str) -> Dict:
        """Detect trading patterns"""
        try:
            price_history = await self.tools.get_price_history(token_address)
            
            patterns = {
                "trend": self._detect_trend(price_history),
                "support_resistance": self._find_support_resistance(price_history),
                "volume_pattern": self._analyze_volume_pattern(price_history)
            }
            
            return patterns
        except Exception as e:
            self.logger.error(f"Pattern detection failed: {e}")
            return {}

    def _calculate_distribution_score(self, holdings: List[Dict], total_supply: float) -> float:
        """Calculate token distribution score (0-100)"""
        if not holdings or total_supply == 0:
            return 0

        # Calculate Gini coefficient
        cumulative_holdings = 0
        gini_numerator = 0
        n = len(holdings)

        for index, holding in enumerate(holdings, 1):
            balance = holding.get("balance", 0)
            cumulative_holdings += balance
            gini_numerator += (n + 1 - index) * balance

        gini = (2 * gini_numerator) / (n * cumulative_holdings) - (n + 1) / n
        
        # Convert to 0-100 score (inverse of Gini, as lower Gini means better distribution)
        return (1 - gini) * 100

    def _calculate_price_impact(self, pool_info: Dict) -> float:
        """Calculate price impact for standard trade sizes"""
        try:
            token0_reserve = float(pool_info.get("token0_reserve", 0))
            token1_reserve = float(pool_info.get("token1_reserve", 0))
            
            if token0_reserve == 0 or token1_reserve == 0:
                return float('inf')
            
            # Calculate price impact for 1% of pool liquidity
            trade_size = token0_reserve * 0.01
            constant_product = token0_reserve * token1_reserve
            new_token1_reserve = constant_product / (token0_reserve + trade_size)
            price_impact = (token1_reserve - new_token1_reserve) / token1_reserve * 100
            
            return price_impact
        except Exception as e:
            self.logger.error(f"Price impact calculation failed: {e}")
            return float('inf')

    def _detect_trend(self, price_history: List[Dict]) -> str:
        """Detect price trend"""
        if not price_history or len(price_history) < 2:
            return "insufficient_data"

        prices = [p.get("price", 0) for p in price_history]
        ma_20 = sum(prices[-20:]) / min(20, len(prices))
        ma_50 = sum(prices[-50:]) / min(50, len(prices))
        
        current_price = prices[-1]
        
        if current_price > ma_20 and ma_20 > ma_50:
            return "uptrend"
        elif current_price < ma_20 and ma_20 < ma_50:
            return "downtrend"
        else:
            return "sideways"