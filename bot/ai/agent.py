from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from langchain_openai import ChatOpenAI
from typing import Dict, Optional, List
import asyncio
import logging
from datetime import datetime

class CDPAgent:
    def __init__(self, config: Dict):
        """Initialize CDP Agent with configuration"""
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

    async def initialize(self) -> Dict:
        """Initialize CDP wallet and tools"""
        try:
            self.wallet_data = self.cdp.export_wallet()
            await self._initialize_tools()
            return {"status": "success", "wallet": self.wallet_data}
        except Exception as e:
            self.logger.error(f"Failed to initialize CDP agent: {e}")
            return {"status": "error", "message": str(e)}

    async def analyze_token(self, token_address: str, tier: str = "basic") -> Dict:
        """Analyze token with tier-based depth"""
        try:
            analysis = {"timestamp": datetime.utcnow().isoformat()}
            
            # Base analysis for all tiers
            base_info = await self.tools.get_token_info(token_address)
            analysis["basic"] = {
                "price": base_info.get("price"),
                "volume": base_info.get("volume"),
                "market_cap": base_info.get("market_cap"),
                "holders": base_info.get("holders")
            }
            
            # Enhanced analysis for higher tiers
            if tier in ["pro", "premium"]:
                analysis["advanced"] = await self._get_advanced_analysis(token_address)
                
                if tier == "premium":
                    analysis["premium"] = await self._get_premium_analysis(
                        token_address, 
                        analysis["basic"],
                        analysis["advanced"]
                    )
            
            return {"status": "success", "analysis": analysis}
        except Exception as e:
            self.logger.error(f"Token analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    async def deploy_token(self, params: Dict) -> Dict:
        """Deploy new token using CDP"""
        try:
            deployment = await self.tools.deploy_token({
                "name": params["name"],
                "symbol": params["symbol"],
                "initial_supply": params["supply"],
                "uri": params.get("uri", "")
            })
            
            return {
                "status": "success",
                "contract_address": deployment.get("address"),
                "transaction_hash": deployment.get("tx_hash")
            }
        except Exception as e:
            self.logger.error(f"Token deployment failed: {e}")
            return {"status": "error", "message": str(e)}

    async def create_liquidity_pool(self, token_address: str, eth_amount: float) -> Dict:
        """Create liquidity pool for token"""
        try:
            pool = await self.tools.create_liquidity_pool({
                "token_address": token_address,
                "eth_amount": eth_amount
            })
            
            return {
                "status": "success",
                "pool_address": pool.get("address"),
                "transaction_hash": pool.get("tx_hash")
            }
        except Exception as e:
            self.logger.error(f"Liquidity pool creation failed: {e}")
            return {"status": "error", "message": str(e)}

    async def monitor_price(self, token_address: str, alert_conditions: Dict) -> Dict:
        """Set up price monitoring with alerts"""
        try:
            if token_address in self.monitoring_tokens:
                return {"status": "error", "message": "Already monitoring this token"}
            
            self.monitoring_tokens.add(token_address)
            self.active_alerts[token_address] = alert_conditions
            
            # Start monitoring task
            asyncio.create_task(self._price_monitor_loop(token_address))
            
            return {"status": "success", "message": "Price monitoring started"}
        except Exception as e:
            self.logger.error(f"Failed to start price monitoring: {e}")
            return {"status": "error", "message": str(e)}

    async def get_user_alerts(self, user_id: int) -> Dict:
        """Get active alerts for user"""
        try:
            user_alerts = {}
            for token, conditions in self.active_alerts.items():
                if conditions.get("user_id") == user_id:
                    user_alerts[token] = conditions
            return user_alerts
        except Exception as e:
            self.logger.error(f"Failed to get user alerts: {e}")
            return {}

    async def _initialize_tools(self) -> None:
        """Initialize CDP tools and verify connections"""
        try:
            # Verify wallet
            await self.tools.get_wallet_details()
            
            # Request testnet funds if needed
            if self.config.get("network") == "base-sepolia":
                await self._ensure_testnet_funds()
            
        except Exception as e:
            self.logger.error(f"Tool initialization failed: {e}")
            raise

    async def _ensure_testnet_funds(self) -> None:
        """Ensure wallet has testnet funds"""
        try:
            balance = await self.tools.get_balance()
            if float(balance) < 0.1:  # Less than 0.1 ETH
                await self.tools.request_faucet_funds()
        except Exception as e:
            self.logger.error(f"Failed to ensure testnet funds: {e}")

    async def _get_advanced_analysis(self, token_address: str) -> Dict:
        """Get advanced token analysis"""
        try:
            liquidity = await self.tools.analyze_liquidity(token_address)
            holders = await self.tools.get_token_holders(token_address)
            
            return {
                "liquidity": liquidity.get("total_liquidity"),
                "price_change": liquidity.get("price_change_24h"),
                "holder_distribution": self._analyze_holders(holders),
                "patterns": await self._detect_patterns(token_address)
            }
        except Exception as e:
            self.logger.error(f"Advanced analysis failed: {e}")
            return {}

    async def _get_premium_analysis(self, token_address: str, basic: Dict, advanced: Dict) -> Dict:
        """Get premium token analysis with AI insights"""
        try:
            # Prepare data for AI analysis
            analysis_data = {
                "token_address": token_address,
                "basic_metrics": basic,
                "advanced_metrics": advanced
            }
            
            # Generate AI insights
            signals = await self._generate_trading_signals(analysis_data)
            recommendation = await self._generate_recommendation(analysis_data)
            
            return {
                "signals": signals,
                "recommendation": recommendation,
                "risk_score": await self._calculate_risk_score(analysis_data)
            }
        except Exception as e:
            self.logger.error(f"Premium analysis failed: {e}")
            return {}

    async def _price_monitor_loop(self, token_address: str) -> None:
        """Monitor token price and trigger alerts"""
        while token_address in self.monitoring_tokens:
            try:
                current_price = await self.tools.get_token_price(token_address)
                alerts = self.active_alerts[token_address]
                
                if self._check_alert_conditions(current_price, alerts):
                    await self._trigger_alert(token_address, current_price, alerts)
                
                await asyncio.sleep(self.config.get("monitor_interval", 60))
            except Exception as e:
                self.logger.error(f"Monitor error for {token_address}: {e}")
                await asyncio.sleep(self.config.get("error_retry_interval", 300))

    def _check_alert_conditions(self, current_price: float, alerts: Dict) -> bool:
        """Check if alert conditions are met"""
        last_price = alerts.get("last_price")
        if not last_price:
            alerts["last_price"] = current_price
            return False
        
        price_change = ((current_price - last_price) / last_price) * 100
        return abs(price_change) >= alerts.get("price_threshold", 5.0)

    async def _trigger_alert(self, token_address: str, price: float, alerts: Dict) -> None:
        """Trigger price alert"""
        try:
            user_id = alerts.get("user_id")
            if user_id and self.alert_callback:
                await self.alert_callback(user_id, {
                    "token_address": token_address,
                    "current_price": price,
                    "price_change": ((price - alerts["last_price"]) / alerts["last_price"]) * 100
                })
            alerts["last_price"] = price
        except Exception as e:
            self.logger.error(f"Failed to trigger alert: {e}")