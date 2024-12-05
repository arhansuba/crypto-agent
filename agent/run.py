import time
import json
import asyncio
from typing import Dict, Any, Optional
from swarm import Swarm
from swarm.repl import run_demo_loop
from openai import OpenAI

from utils.logger import get_logger
from utils.config_manager import ConfigManager
from agents import AIEnhancedAgent
from analysis.market_analyzer import MarketAnalyzer
from risk.portfolio_manager import PortfolioManager

logger = get_logger(__name__)

class EnhancedAgentRunner:
    """
    Runner class for managing AI-enhanced CDP agent operations.
    """
    
    def __init__(self, config_path: str):
        self.config = ConfigManager(config_path)
        self.agent = AIEnhancedAgent(self.config.get_agent_config())
        self.client = Swarm()
        self.market_analyzer = MarketAnalyzer(self.config.get_market_config())
        self.portfolio_manager = PortfolioManager(self.config.get_risk_config())
        
    async def process_streaming_response(self, response) -> Optional[Dict[str, Any]]:
        """Process streaming responses from the agent with enhanced logging."""
        content = ""
        last_sender = ""

        try:
            for chunk in response:
                if "sender" in chunk:
                    last_sender = chunk["sender"]
                    logger.info(f"Message from: {last_sender}")

                if "content" in chunk and chunk["content"]:
                    content_chunk = self._format_content(last_sender, chunk["content"])
                    print(content_chunk, end="", flush=True)
                    content += chunk["content"]

                if "tool_calls" in chunk and chunk["tool_calls"]:
                    await self._handle_tool_calls(chunk["tool_calls"], last_sender)

                if "delim" in chunk and chunk["delim"] == "end" and content:
                    print()
                    content = ""

                if "response" in chunk:
                    return chunk["response"]

        except Exception as e:
            logger.error(f"Error processing stream: {str(e)}")
            return None

    async def run_autonomous_mode(self, interval: int = 10):
        """Run agent in autonomous mode with AI-driven decision making."""
        messages = []
        
        logger.info("Starting autonomous AI-enhanced agent...")
        await self._display_initial_status()

        while True:
            try:
                # Get market analysis
                market_state = await self.market_analyzer.get_current_state()
                
                # Generate AI-driven strategy
                strategy = await self.agent.generate_strategy(market_state)
                
                # Validate strategy against risk parameters
                if await self.portfolio_manager.validate_strategy(strategy):
                    # Execute strategy
                    execution_result = await self.agent.execute_ai_trading(
                        strategy_type=strategy["type"],
                        params=strategy["params"]
                    )
                    
                    # Record results and update models
                    await self._record_execution_results(execution_result)
                    
                    # Generate and display performance report
                    await self._display_performance_update(execution_result)

                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Autonomous mode error: {str(e)}")
                await asyncio.sleep(self.config.get("error_retry_interval"))

    async def run_interactive_mode(self):
        """Run agent in interactive mode with AI assistance."""
        logger.info("Starting interactive AI-enhanced agent mode...")
        
        while True:
            try:
                user_input = input("\nEnter command (or 'exit' to quit): ")
                if user_input.lower() == 'exit':
                    break

                # Analyze command with AI
                analyzed_command = await self.agent.analyze_user_command(user_input)
                
                # Execute command with AI optimization
                execution_result = await self.agent.execute_command(analyzed_command)
                
                # Display results
                await self._display_execution_results(execution_result)
                
            except Exception as e:
                logger.error(f"Interactive mode error: {str(e)}")

    async def run_ai_conversation_mode(self):
        """Run agent in AI conversation mode with enhanced analysis."""
        openai_client = OpenAI()
        messages = []

        logger.info("Starting AI conversation mode...")

        while True:
            try:
                # Generate AI response with market context
                market_context = await self.market_analyzer.get_market_context()
                ai_response = await self._generate_ai_response(openai_client, market_context)
                
                # Process AI command through agent
                execution_result = await self.agent.process_ai_command(ai_response)
                
                # Update conversation state
                messages.extend(self._format_conversation(ai_response, execution_result))
                
                # Check for continuation
                if not await self._should_continue_conversation():
                    break
                    
            except Exception as e:
                logger.error(f"AI conversation error: {str(e)}")

    @staticmethod
    def choose_mode() -> str:
        """Enhanced mode selection with validation."""
        while True:
            print("\nAvailable modes:")
            print("1. interactive - Interactive trading mode with AI assistance")
            print("2. autonomous  - Autonomous trading with AI optimization")
            print("3. ai-conversation - AI-guided trading conversation")

            choice = input("\nChoose mode (enter number or name): ").lower().strip()

            mode_map = {
                '1': 'interactive',
                '2': 'autonomous',
                '3': 'ai-conversation',
                'interactive': 'interactive',
                'autonomous': 'autonomous',
                'ai-conversation': 'ai-conversation'
            }

            if choice in mode_map:
                return mode_map[choice]
            print("Invalid choice. Please try again.")

def main():
    """Main entry point with enhanced error handling and logging."""
    try:
        config_path = "config/agent_config.yaml"
        runner = EnhancedAgentRunner(config_path)
        
        mode = runner.choose_mode()
        logger.info(f"Starting agent in {mode} mode")

        mode_map = {
            'interactive': runner.run_interactive_mode,
            'autonomous': runner.run_autonomous_mode,
            'ai-conversation': runner.run_ai_conversation_mode
        }

        asyncio.run(mode_map[mode]())
        
    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()