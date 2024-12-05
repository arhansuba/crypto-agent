from langchain_openai import ChatOpenAI
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from typing import Dict, Any, Optional, List, Union
import asyncio
import logging
from decimal import Decimal

from ai.models.market_predictor import MarketPredictionModel
from ai.training.trainer import AIModelTrainer
from analysis.market_analyzer import MarketAnalyzer
from analysis.pattern_detector import PatternDetector
from utils.logger import get_logger
from risk.portfolio_manager import PortfolioManager

logger = get_logger(__name__)

class EnhancedCDPAgent:
    """
    An enhanced CDP agent that combines CDP's blockchain capabilities with AI-powered
    analysis and automated trading features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced CDP agent with AI components."""
        self.config = config
        
        # Initialize CDP Core Components
        self.llm = ChatOpenAI(model=config.get("llm_model", "gpt-4o-mini"))
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        self.tools = self.toolkit.get_tools()
        
        # Initialize Additional Tools
        self._initialize_additional_tools()
        
        # Initialize AI Components
        self.market_predictor = MarketPredictionModel(config.get("market_model_config"))
        self.market_analyzer = MarketAnalyzer(config.get("market_analysis_config"))
        self.pattern_detector = PatternDetector(config.get("pattern_detection_config"))
        
        # Initialize Risk Management
        self.portfolio_manager = PortfolioManager(config.get("risk_config"))
        
        # Initialize Memory and State
        self.memory = MemorySaver()
        self.conversation_config = {"configurable": {"thread_id": "enhanced_cdp_agent"}}

    def _initialize_additional_tools(self):
        """Initialize additional CDP tools and capabilities."""
        additional_tools = []
        
        # Add market analysis tool
        market_analysis_tool = self._create_market_analysis_tool()
        additional_tools.append(market_analysis_tool)
        
        # Add trading strategy tool
        trading_strategy_tool = self._create_trading_strategy_tool()
        additional_tools.append(trading_strategy_tool)
        
        # Add risk management tool
        risk_management_tool = self._create_risk_management_tool()
        additional_tools.append(risk_management_tool)
        
        # Combine all tools
        self.tools.extend(additional_tools)

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the agent and its components."""
        try:
            # Initialize CDP wallet
            wallet_data = self.cdp.export_wallet()
            
            # Initialize AI models
            await self._initialize_models()
            
            # Create the React agent
            self.agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                checkpointer=self.memory,
                state_modifier=self._get_agent_instructions()
            )
            
            return {
                "status": "success",
                "wallet_data": wallet_data,
                "initialized_tools": [tool.name for tool in self.tools]
            }
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def process_command(self, command: str) -> Dict[str, Any]:
        """Process a command with AI-enhanced analysis."""
        try:
            # Analyze command with market context
            market_state = await self.market_analyzer.get_current_state()
            analysis = await self.market_predictor.analyze_command(
                command=command,
                market_state=market_state
            )
            
            # Execute command with AI optimization
            response = await self.execute_command(
                command=command,
                analysis=analysis
            )
            
            return {
                "status": "success",
                "response": response,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Command processing failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def execute_command(
        self,
        command: str,
        analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a command through the CDP agent with AI optimization."""
        try:
            for chunk in self.agent.stream(
                {"messages": [HumanMessage(content=command)]},
                self.conversation_config
            ):
                if "agent" in chunk:
                    message = chunk["agent"]["messages"][0].content
                    await self._process_agent_message(message, analysis)
                elif "tools" in chunk:
                    result = await self._process_tool_execution(
                        chunk["tools"],
                        analysis
                    )
                    
            return result
            
        except Exception as e:
            logger.error(f"Command execution failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _get_agent_instructions(self) -> str:
        """Get enhanced agent instructions."""
        return """You are an advanced agent that combines CDP capabilities with AI-powered analysis.
        You can interact with the Base blockchain using CDP AgentKit, create and manage wallets,
        deploy tokens, and perform transactions. You also have AI-enhanced capabilities for
        market analysis, risk management, and trading optimization. You should always consider
        market conditions and risk parameters before executing trades."""

    async def _process_agent_message(
        self,
        message: str,
        analysis: Optional[Dict[str, Any]]
    ):
        """Process and enhance agent messages with AI insights."""
        if analysis:
            enhanced_message = await self.market_predictor.enhance_response(
                message=message,
                analysis=analysis
            )
            return enhanced_message
        return message
    
    async def handle_market_operations(self, operation_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle market operations with AI-enhanced analysis and risk management.
        
        Args:
            operation_type: Type of market operation (e.g., 'trade', 'liquidity', 'token_launch')
            params: Operation parameters and constraints
            
        Returns:
            Dict containing operation results and analysis
        """
        try:
            # Perform market analysis
            market_analysis = await self.market_analyzer.analyze_market_conditions()
            sentiment_data = await self.pattern_detector.analyze_market_patterns()
            
            # Generate operation strategy
            strategy = await self.market_predictor.generate_operation_strategy(
                operation_type=operation_type,
                market_analysis=market_analysis,
                sentiment_data=sentiment_data,
                params=params
            )
            
            # Validate with risk management
            risk_assessment = await self.portfolio_manager.assess_operation_risk(
                strategy=strategy,
                current_market_state=market_analysis
            )
            
            if not risk_assessment["approved"]:
                return {
                    "status": "rejected",
                    "reason": risk_assessment["reason"],
                    "analysis": risk_assessment["analysis"]
                }
            
            # Execute operation through CDP
            result = await self._execute_cdp_operation(
                operation_type=operation_type,
                strategy=strategy,
                params=params
            )
            
            # Record results for model training
            await self._record_operation_result(result)
            
            return {
                "status": "success",
                "result": result,
                "analysis": {
                    "market": market_analysis,
                    "sentiment": sentiment_data,
                    "risk": risk_assessment
                }
            }
            
        except Exception as e:
            logger.error(f"Market operation failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def _execute_cdp_operation(
        self,
        operation_type: str,
        strategy: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute CDP operation with optimized parameters.
        """
        try:
            if operation_type == "trade":
                return await self._execute_trade(strategy, params)
            elif operation_type == "liquidity":
                return await self._manage_liquidity(strategy, params)
            elif operation_type == "token_launch":
                return await self._launch_token(strategy, params)
            else:
                raise ValueError(f"Unsupported operation type: {operation_type}")
                
        except Exception as e:
            logger.error(f"CDP operation execution failed: {str(e)}")
            raise

    async def monitor_market_conditions(self) -> None:
        """
        Continuously monitor market conditions and trigger actions when needed.
        """
        while True:
            try:
                # Analyze current market state
                market_state = await self.market_analyzer.get_current_state()
                
                # Generate market insights
                insights = await self.market_predictor.generate_market_insights(
                    market_state=market_state
                )
                
                # Check for action triggers
                if insights["requires_action"]:
                    await self._handle_market_action(
                        insights=insights,
                        market_state=market_state
                    )
                
                await asyncio.sleep(self.config.get("market_check_interval", 60))
                
            except Exception as e:
                logger.error(f"Market monitoring error: {str(e)}")
                await asyncio.sleep(self.config.get("error_retry_interval", 300))