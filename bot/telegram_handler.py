from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)
import logging
from typing import Dict, Optional
from .ai.agent import CDPTradingAgent
from .services.subscription import SubscriptionService
from .services.user_manager import UserManager

class TelegramBot:
    def __init__(self, config: Dict):
        """Initialize Telegram bot with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.agent = CDPTradingAgent(config)
        self.subscription_service = SubscriptionService()
        self.user_manager = UserManager()

    async def initialize(self) -> None:
        """Initialize bot and its components"""
        # Initialize CDP agent
        await self.agent.initialize()
        
        # Create application
        self.application = Application.builder().token(self.config["telegram_token"]).build()
        
        # Add handlers
        self._register_handlers()
        
        self.logger.info("Telegram bot initialized successfully")

    def _register_handlers(self) -> None:
        """Register all command and message handlers"""
        # Core commands
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("help", self.cmd_help))
        
        # Analysis commands
        self.application.add_handler(CommandHandler("analyze", self.cmd_analyze))
        self.application.add_handler(CommandHandler("monitor", self.cmd_monitor))
        self.application.add_handler(CommandHandler("alerts", self.cmd_alerts))
        
        # Subscription commands
        self.application.add_handler(CommandHandler("subscribe", self.cmd_subscribe))
        self.application.add_handler(CommandHandler("status", self.cmd_status))
        
        # General message handler
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Error handler
        self.application.add_error_handler(self.error_handler)

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command"""
        user_id = update.effective_user.id
        await self.user_manager.register_user(user_id)
        
        welcome_message = (
            "🤖 Welcome to the CDP Trading Bot!\n\n"
            "I can help you analyze and monitor crypto tokens on the Base network.\n\n"
            "Available commands:\n"
            "/analyze <token_address> - Analyze a token\n"
            "/monitor <token_address> - Start price monitoring\n"
            "/alerts - View your active alerts\n"
            "/subscribe - View subscription plans\n"
            "/help - Show detailed help\n\n"
            "Start with /analyze to check your first token!"
        )
        
        await update.message.reply_text(welcome_message)

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command"""
        help_text = (
            "📚 Command Reference:\n\n"
            "Analysis Commands:\n"
            "/analyze <token_address> - Get detailed token analysis\n"
            "/monitor <token_address> - Start price monitoring\n"
            "/alerts - View and manage your alerts\n\n"
            "Subscription Commands:\n"
            "/subscribe - View and manage subscription\n"
            "/status - Check your current plan\n\n"
            "Need more help? Contact our support!"
        )
        
        await update.message.reply_text(help_text)

    async def cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /analyze command"""
        if not context.args:
            await update.message.reply_text("Please provide a token address.\nUsage: /analyze <token_address>")
            return

        token_address = context.args[0]
        user_id = update.effective_user.id
        
        # Get user's subscription tier
        tier = await self.subscription_service.get_user_tier(user_id)
        
        try:
            # Call CDP agent to analyze token
            analysis = await self.agent.analyze_token(token_address, tier)
            
            if analysis["status"] == "success":
                response = self._format_analysis_response(analysis["analysis"], tier)
                await update.message.reply_text(response)
            else:
                await update.message.reply_text(f"Analysis failed: {analysis['message']}")
        
        except Exception as e:
            self.logger.error(f"Error in analyze command: {e}")
            await update.message.reply_text("Sorry, an error occurred while analyzing the token.")

    async def cmd_monitor(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /monitor command"""
        if not context.args:
            await update.message.reply_text("Please provide a token address.\nUsage: /monitor <token_address>")
            return

        token_address = context.args[0]
        user_id = update.effective_user.id
        
        try:
            # Start monitoring with CDP agent
            result = await self.agent.monitor_price(
                token_address,
                {"user_id": user_id}
            )
            
            if result["status"] == "success":
                await update.message.reply_text(
                    f"🔍 Now monitoring {token_address}\n"
                    "You'll receive alerts for significant price changes!"
                )
            else:
                await update.message.reply_text(f"Failed to start monitoring: {result['message']}")
                
        except Exception as e:
            self.logger.error(f"Error in monitor command: {e}")
            await update.message.reply_text("Sorry, an error occurred while setting up monitoring.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle non-command messages"""
        await update.message.reply_text(
            "Please use commands to interact with me. Type /help to see available commands."
        )

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors in the bot"""
        self.logger.error(f"Error: {context.error} - {update}")
        
        if update and isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(
                "Sorry, an error occurred. Please try again later."
            )

    def _format_analysis_response(self, analysis: Dict, tier: str) -> str:
        """Format token analysis response based on user's tier"""
        response = ["📊 Token Analysis:\n"]
        
        # Basic info for all tiers
        basic = analysis.get("basic", {})
        response.append(f"Price: {basic.get('price', 'N/A')}")
        response.append(f"24h Volume: {basic.get('volume', 'N/A')}")
        
        # Advanced info for pro/premium tiers
        if tier in ["pro", "premium"] and "advanced" in analysis:
            advanced = analysis["advanced"]
            response.append("\n🔍 Advanced Analysis:")
            response.append(f"Liquidity: {advanced.get('liquidity', 'N/A')}")
            response.append(f"Pattern: {advanced.get('patterns', 'N/A')}")
        
        # Premium insights
        if tier == "premium" and "premium" in analysis:
            premium = analysis["premium"]
            response.append("\n🎯 Trading Signals:")
            response.append(f"Recommendation: {premium.get('recommendation', 'N/A')}")
        
        return "\n".join(response)

    async def run(self) -> None:
        """Start the bot"""
        self.logger.info("Starting bot...")
        await self.application.initialize()
        await self.application.start()
        await self.application.run_polling()