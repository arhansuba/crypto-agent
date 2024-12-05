import os
import sys
import time
import json
import asyncio
from typing import Dict, Any, Optional
from openai import OpenAI

# Add project root to Python path to ensure proper imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Local imports
from utils.logger import get_logger
from utils.config_manager import ConfigManager
from agents import AIEnhancedAgent
from analysis.market_analyzer import MarketAnalyzer
from risk.portfolio_manager import PortfolioManager
# Add new imports while keeping existing ones
from cdp_core.base_agent import BaseCDPAgent
from core.brain import Brain
from execution.cdp_executor import CDPExecutor
from strategies.arbitrage.cdp_dex_monitor import CDPDEXMonitor
from bot.telegram_handler import TelegramHandler

logger = get_logger(__name__)

class EnhancedAgentRunner:
    """
    Runner class for managing AI-enhanced CDP agent operations.
    """
    
    def __init__(self, config_path: str):
        self.config = ConfigManager(config_path)
        self.agent = None
        self.market_analyzer = None
        self.portfolio_manager = None
        self.components_initialized = False
        
        # Initialize components after proper setup
        self._initialize_base_components()

    def _initialize_base_components(self):
        """Initialize base components with proper error handling."""
        try:
            self.agent = AIEnhancedAgent(self.config.get_agent_config())
            self.market_analyzer = MarketAnalyzer(self.config.get_market_config())
            self.portfolio_manager = PortfolioManager(self.config.get_risk_config())
            
            # Initialize CDP components
            self.base_cdp_agent = BaseCDPAgent(self.config.get_agent_config())
            self.brain = Brain()
            self.executor = CDPExecutor()
            self.dex_monitor = CDPDEXMonitor(self.base_cdp_agent)
            self.telegram_handler = TelegramHandler(self)
            
            # Initialize strategy monitoring
            self._init_strategy_monitoring()
            
            logger.info("Base components initialized successfully")
            
        except Exception as e:
            logger.error(f"Base component initialization failed: {str(e)}")
            raise

    async def initialize_all_components(self):
        """Initialize all components that require async initialization."""
        if self.components_initialized:
            return
            
        try:
            await self.base_cdp_agent.initialize()
            await self.telegram_handler.start()
            await self.dex_monitor.start_monitoring()
            await self.market_analyzer.initialize()
            
            # Initialize strategy monitors
            for monitor in self.strategy_monitors.values():
                await monitor.start()
                
            self.components_initialized = True
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization error: {str(e)}")
            raise

    # Keep all your existing methods unchanged

    async def start(self, mode: str):
        """Start the agent with the specified mode after ensuring proper initialization."""
        try:
            await self.initialize_all_components()
            
            mode_map = {
                'interactive': self.run_interactive_mode,
                'autonomous': self.run_autonomous_mode,
                'ai-conversation': self.run_ai_conversation_mode
            }
            
            if mode not in mode_map:
                raise ValueError(f"Invalid mode: {mode}")
                
            logger.info(f"Starting agent in {mode} mode")
            await mode_map[mode]()
            
        except Exception as e:
            logger.error(f"Agent start failed: {str(e)}")
            raise

def main():
    """Main entry point with enhanced error handling and logging."""
    try:
        config_path = "config/agent_config.yaml"
        runner = EnhancedAgentRunner(config_path)
        
        mode = runner.choose_mode()
        asyncio.run(runner.start(mode))
        
    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        sys.exit(0)

if __name__ == "__main__":
    runner = EnhancedAgentRunner(config_path="path/to/config.yaml")
    runner.initialize()
    runner.run()