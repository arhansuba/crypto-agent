from typing import Dict, List, Union, Optional
from datetime import datetime

class ResponseFormatter:
    """Handles formatting of all bot responses"""
    
    @staticmethod
    def format_token_analysis(analysis: Dict, tier: str) -> str:
        """Format token analysis response"""
        sections = []
        
        # Header
        sections.append("🔎 Token Analysis Report")
        sections.append("=" * 25)
        
        # Basic Analysis (All Tiers)
        basic = analysis.get("basic", {})
        sections.extend([
            "📊 Basic Metrics",
            f"• Price: ${basic.get('price', 'N/A'):,.4f}",
            f"• 24h Volume: ${basic.get('volume', 'N/A'):,.2f}",
            f"• Market Cap: ${basic.get('market_cap', 'N/A'):,.2f}",
            ""
        ])
        
        # Advanced Analysis (Pro & Premium)
        if tier in ["pro", "premium"] and "advanced" in analysis:
            advanced = analysis["advanced"]
            sections.extend([
                "🔍 Advanced Analysis",
                f"• Liquidity: ${advanced.get('liquidity', 'N/A'):,.2f}",
                f"• 24h Change: {advanced.get('price_change', 'N/A')}%",
                "• Pattern: " + advanced.get('patterns', 'No patterns detected'),
                ""
            ])
        
        # Premium Analysis
        if tier == "premium" and "premium" in analysis:
            premium = analysis["premium"]
            sections.extend([
                "🎯 AI Insights",
                f"• Signal: {premium.get('signals', 'N/A')}",
                f"• Strength: {premium.get('signal_strength', 'N/A')}",
                f"• Recommendation: {premium.get('recommendation', 'N/A')}",
                ""
            ])
        
        return "\n".join(sections)
    
    @staticmethod
    def format_alert_message(alert: Dict) -> str:
        """Format price alert message"""
        return (
            "🚨 Price Alert!\n\n"
            f"Token: {alert.get('token_symbol', 'Unknown')}\n"
            f"Price: ${alert.get('current_price', 0):,.4f}\n"
            f"Change: {alert.get('price_change', 0):+.2f}%\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
    
    @staticmethod
    def format_subscription_plans(plans: List[Dict], current_tier: str) -> str:
        """Format subscription plans message"""
        sections = ["📱 Available Subscription Plans\n"]
        
        for plan in plans:
            # Plan header
            if plan['tier'] == current_tier:
                sections.append(f"✅ {plan['tier'].upper()} (Current Plan)")
            else:
                sections.append(f"💠 {plan['tier'].upper()}")
                
            # Plan details
            sections.extend([
                f"Price: ${plan['price']}/month",
                "Features:"
            ])
            
            # Plan features
            for feature in plan['features']:
                sections.append(f"  • {feature}")
            sections.append("")  # Empty line between plans
            
        return "\n".join(sections)
    
    @staticmethod
    def format_monitor_status(monitors: List[Dict]) -> str:
        """Format monitoring status message"""
        if not monitors:
            return "📊 No active monitors. Use /monitor to start tracking tokens!"
        
        sections = ["📊 Active Monitors\n"]
        
        for monitor in monitors:
            sections.extend([
                f"Token: {monitor.get('symbol', 'Unknown')}",
                f"Price: ${monitor.get('current_price', 0):,.4f}",
                f"Alert Threshold: {monitor.get('alert_threshold', 0)}%",
                f"Started: {monitor.get('start_time', 'Unknown')}\n"
            ])
            
        return "\n".join(sections)
    
    @staticmethod
    def format_error_message(error: str, suggestion: Optional[str] = None) -> str:
        """Format error message with optional suggestion"""
        message = f"❌ Error: {error}"
        if suggestion:
            message += f"\n💡 Suggestion: {suggestion}"
        return message
    
    @staticmethod
    def format_success_message(action: str) -> str:
        """Format success message"""
        return f"✅ Success: {action}"
    
    @staticmethod
    def format_welcome_message(username: str) -> str:
        """Format welcome message for new users"""
        return (
            f"👋 Welcome {username}!\n\n"
            "I'm your crypto trading assistant. Here's what I can do:\n\n"
            "🔍 Token Analysis:\n"
            "• /analyze <token_address> - Analyze any token\n"
            "• /monitor <token_address> - Track price changes\n"
            "• /alerts - View your active alerts\n\n"
            "💎 Account & Settings:\n"
            "• /subscribe - View subscription plans\n"
            "• /status - Check your account status\n"
            "• /help - Get detailed help\n\n"
            "Start with /analyze to check your first token!"
        )
    
    @staticmethod
    def format_help_message() -> str:
        """Format help message"""
        return (
            "📚 Command Reference\n\n"
            "Analysis Commands:\n"
            "• /analyze <token> - Get token analysis\n"
            "• /monitor <token> - Start price tracking\n"
            "• /alerts - View active alerts\n\n"
            "Account Commands:\n"
            "• /subscribe - View/change subscription\n"
            "• /status - Account status\n"
            "• /settings - Bot settings\n\n"
            "Need help? Contact @support"
        )

    @staticmethod
    def format_status_message(user_data: Dict) -> str:
        """Format user status message"""
        return (
            "📊 Your Account Status\n\n"
            f"Plan: {user_data.get('tier', 'Free')}\n"
            f"Analyses Today: {user_data.get('analyses_today', 0)}/{user_data.get('daily_limit', 3)}\n"
            f"Active Monitors: {user_data.get('active_monitors', 0)}/{user_data.get('monitor_limit', 1)}\n"
            f"Member Since: {user_data.get('joined_date', 'Unknown')}\n\n"
            "Use /subscribe to upgrade your plan!"
        )