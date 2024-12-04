from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from langchain_openai import ChatOpenAI
from typing import Dict, Optional, List
import asyncio
import logging

class CDPTradingAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        self.tools = self.toolkit.get_tools()
        self.logger = logging.getLogger(__name__)
        
        # Initialize state
        self.monitoring_tokens = set()
        self.active_alerts = {}
        self.wallet_data = None

    async def initialize(self):
        """Initialize agent and load wallet"""
        try:
            self.wallet_data = self.cdp.export_wallet()
            return {"status": "success", "wallet": self.wallet_data}
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {e}")
            return {"status": "error", "message": str(e)}

    async def analyze_token(self, token_address: str, tier: str = "basic") -> Dict:
        """Analyze token based on subscription tier"""
        try:
            analysis = {}
            
            # Basic analysis for all tiers
            analysis["basic"] = await self.tools.get_token_info(token_address)
            
            if tier in ["pro", "premium"]:
                # Add advanced analysis
                liquidity = await self.tools.analyze_liquidity(token_address)
                patterns = await self.analyze_patterns(token_address)
                analysis["advanced"] = {
                    "liquidity": liquidity,
                    "patterns": patterns
                }
                
                if tier == "premium":
                    # Add AI-powered insights
                    signals = await self.generate_trading_signals(token_address)
                    analysis["premium"] = {
                        "signals": signals,
                        "recommendation": await self._generate_recommendation(analysis)
                    }
            
            return {"status": "success", "analysis": analysis}
        except Exception as e:
            self.logger.error(f"Analysis failed for {token_address}: {e}")
            return {"status": "error", "message": str(e)}

    async def monitor_price(self, token_address: str, alert_conditions: Dict):
        """Set up price monitoring and alerts"""
        if token_address not in self.monitoring_tokens:
            self.monitoring_tokens.add(token_address)
            self.active_alerts[token_address] = alert_conditions
            
            asyncio.create_task(self._price_monitor_loop(token_address))
            return {"status": "success", "message": "Monitoring started"}
        return {"status": "error", "message": "Already monitoring this token"}

    async def execute_trade(self, params: Dict) -> Dict:
        """Execute trade with position sizing and risk management"""
        try:
            # Validate parameters
            if not self._validate_trade_params(params):
                return {"status": "error", "message": "Invalid trade parameters"}
            
            # Calculate position size
            position_size = await self._calculate_position_size(params)
            
            # Execute via CDP
            tx = await self.tools.execute_transaction({
                'type': params['type'],
                'amount': position_size,
                'token_address': params['token_address'],
                'slippage': params.get('slippage', 0.5)
            })
            
            return {"status": "success", "transaction": tx}
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            return {"status": "error", "message": str(e)}

    async def _price_monitor_loop(self, token_address: str):
        """Internal price monitoring loop"""
        while token_address in self.monitoring_tokens:
            try:
                price = await self.tools.get_token_price(token_address)
                alerts = self.active_alerts[token_address]
                
                if self._check_alert_conditions(price, alerts):
                    await self._trigger_alert(token_address, price)
                
                await asyncio.sleep(self.config.get('monitor_interval', 60))
            except Exception as e:
                self.logger.error(f"Monitor error for {token_address}: {e}")
                await asyncio.sleep(self.config.get('error_retry_interval', 300))

    async def _generate_recommendation(self, analysis: Dict) -> Dict:
        """Generate trading recommendation using AI"""
        try:
            prompt = self._create_analysis_prompt(analysis)
            response = await self.llm.agenerate([prompt])
            return self._parse_recommendation(response.text)
        except Exception as e:
            self.logger.error(f"Failed to generate recommendation: {e}")
            return {"error": str(e)}

    def _validate_trade_params(self, params: Dict) -> bool:
        """Validate trading parameters"""
        required = ['type', 'token_address', 'amount']
        return all(k in params for k in required)

    async def _calculate_position_size(self, params: Dict) -> float:
        """Calculate safe position size based on risk parameters"""
        balance = await self.tools.get_balance()
        risk_per_trade = self.config.get('risk_per_trade', 0.02)
        return float(balance) * risk_per_trade * params.get('risk_multiplier', 1.0)