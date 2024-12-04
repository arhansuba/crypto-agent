from typing import Dict, Optional, List, Tuple
from telegram import Update
from telegram.ext import ContextTypes
import logging
from .ai.agent import CDPTradingAgent
from .services.subscription import SubscriptionService
from .services.user_manager import UserManager

class CommandProcessor:
    def __init__(self, config: Dict):
        """Initialize command processor with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.agent = CDPTradingAgent(config)
        self.subscription_service = SubscriptionService()
        self.user_manager = UserManager()

    async def process_analyze_command(self, update: Update, args: List[str]) -> Tuple[bool, str]:
        """Process the analyze command"""
        if not args:
            return False, "Please provide a token address.\nUsage: /analyze <token_address>"

        token_address = args[0]
        user_id = update.effective_user.id
        
        try:
            # Verify user subscription
            tier = await self.subscription_service.get_user_tier(user_id)
            daily_limit = await self.subscription_service.get_daily_analysis_limit(user_id)
            
            if not await self.subscription_service.can_analyze(user_id):
                return False, f"Daily analysis limit reached. Current plan: {tier}\nUpgrade for more analyses!"
            
            # Perform analysis
            analysis = await self.agent.analyze_token(token_address, tier)
            
            if analysis["status"] == "success":
                # Update usage count
                await self.subscription_service.record_analysis(user_id)
                return True, self._format_analysis(analysis["analysis"], tier)
            else:
                return False, f"Analysis failed: {analysis['message']}"
                
        except Exception as e:
            self.logger.error(f"Error processing analyze command: {e}")
            return False, "An error occurred while analyzing the token."

    async def process_monitor_command(self, update: Update, args: List[str]) -> Tuple[bool, str]:
        """Process the monitor command"""
        if not args:
            return False, "Please provide a token address.\nUsage: /monitor <token_address>"

        token_address = args[0]
        user_id = update.effective_user.id
        
        try:
            # Check monitoring limits
            if not await self.subscription_service.can_add_monitor(user_id):
                return False, "Monitor limit reached. Upgrade your plan for more monitoring slots!"
            
            # Set up monitoring
            alert_conditions = await self._get_default_alert_conditions(user_id)
            result = await self.agent.monitor_price(token_address, alert_conditions)
            
            if result["status"] == "success":
                await self.subscription_service.record_monitor(user_id, token_address)
                return True, "🔍 Monitoring activated! You'll receive alerts for significant price changes."
            else:
                return False, f"Failed to start monitoring: {result['message']}"
                
        except Exception as e:
            self.logger.error(f"Error processing monitor command: {e}")
            return False, "An error occurred while setting up monitoring."

    async def process_subscribe_command(self, update: Update) -> Tuple[bool, str]:
        """Process the subscribe command"""
        try:
            user_id = update.effective_user.id
            current_tier = await self.subscription_service.get_user_tier(user_id)
            plans = await self.subscription_service.get_available_plans()
            
            response = ["📊 Subscription Plans:\n"]
            
            for plan in plans:
                if plan["tier"] == current_tier:
                    response.append(f"➡️ {plan['tier'].upper()} (Current Plan)")
                else:
                    response.append(f"💠 {plan['tier'].upper()}")
                response.append(f"Price: ${plan['price']}/month")
                response.append("Features:")
                for feature in plan["features"]:
                    response.append(f"  • {feature}")
                response.append("")
            
            response.append("\nTo upgrade, use /subscribe <plan_name>")
            return True, "\n".join(response)
            
        except Exception as e:
            self.logger.error(f"Error processing subscribe command: {e}")
            return False, "An error occurred while fetching subscription information."

    async def process_alerts_command(self, update: Update) -> Tuple[bool, str]:
        """Process the alerts command"""
        try:
            user_id = update.effective_user.id
            alerts = await self.agent.get_user_alerts(user_id)
            
            if not alerts:
                return True, "You don't have any active alerts. Use /monitor to set up price alerts!"
            
            response = ["🚨 Your Active Alerts:\n"]
            for token, conditions in alerts.items():
                response.append(f"Token: {token}")
                response.append(f"Price Alerts: {conditions.get('price_alerts', 'None')}")
                response.append(f"Trend Alerts: {conditions.get('trend_alerts', 'None')}")
                response.append("")
            
            return True, "\n".join(response)
            
        except Exception as e:
            self.logger.error(f"Error processing alerts command: {e}")
            return False, "An error occurred while fetching alerts."

    async def _get_default_alert_conditions(self, user_id: int) -> Dict:
        """Get default alert conditions based on user's subscription"""
        tier = await self.subscription_service.get_user_tier(user_id)
        
        conditions = {
            "price_change": 5.0,  # 5% price change
            "volume_change": 20.0,  # 20% volume change
            "check_interval": 300,  # 5 minutes
        }
        
        if tier in ["pro", "premium"]:
            conditions.update({
                "price_change": 3.0,  # More sensitive
                "check_interval": 60,  # 1 minute
                "trend_detection": True
            })
            
        if tier == "premium":
            conditions.update({
                "ai_analysis": True,
                "smart_alerts": True
            })
            
        return conditions

    def _format_analysis(self, analysis: Dict, tier: str) -> str:
        """Format analysis results based on subscription tier"""
        response = ["📊 Token Analysis:\n"]
        
        # Basic metrics
        basic = analysis.get("basic", {})
        response.extend([
            f"💰 Price: ${basic.get('price', 'N/A')}",
            f"📈 24h Volume: ${basic.get('volume', 'N/A')}",
            f"💎 Market Cap: ${basic.get('market_cap', 'N/A')}"
        ])
        
        # Pro/Premium features
        if tier in ["pro", "premium"] and "advanced" in analysis:
            advanced = analysis["advanced"]
            response.extend([
                "\n🔍 Advanced Analysis:",
                f"💧 Liquidity: ${advanced.get('liquidity', 'N/A')}",
                f"📊 Pattern: {advanced.get('patterns', 'None detected')}"
            ])
        
        # Premium features
        if tier == "premium" and "premium" in analysis:
            premium = analysis["premium"]
            response.extend([
                "\n🎯 AI Insights:",
                f"Signal: {premium.get('signals', 'N/A')}",
                f"Recommendation: {premium.get('recommendation', 'N/A')}"
            ])
        
        return "\n".join(response)