"""
Core implementation of the AI Crypto Agent using CDP Agentkit and LangChain.
This module handles the primary agent logic, decision-making, and blockchain interactions.
"""

import os
import json
import logging
import asyncio
import sys
from typing import Dict, Tuple, Any
from datetime import datetime
from enum import Enum

import yaml

from agent.core.memory import AgentState
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langgraph.prebuilt import create_react_agent
from strategies.arbitrage.cdp_dex_monitor import CDPDEXMonitor
from strategies.token_launch.launch_detector import LaunchDetector
from execution.cdp_executor import CDPExecutor
from risk.portfolio_manager import PortfolioManager
from social.sentiment import SentimentAnalyzer


class AgentMode(Enum):
    AUTONOMOUS = "autonomous"
    INTERACTIVE = "interactive"
    ANALYSIS = "analysis"
    STRATEGY_TESTING = "strategy_testing"
    PORTFOLIO_MANAGEMENT = "portfolio_management"


class AgentCore:
    """
    Core implementation of the AI Crypto Agent with advanced decision-making
    and blockchain interaction capabilities.
    """
    
    def __init__(self, config_path: str):
        """Initialize the agent with configuration and required components"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize CDP components
        self.cdp_wrapper = self._initialize_cdp()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp_wrapper)
        
        # Setup AI components
        self.llm = self._initialize_llm()
        self.memory = self._initialize_memory()
        self.agent = self._setup_agent()
        
        # Initialize agent state
        self.state = self._initialize_state()
        
        # Track operation mode
        self.mode = AgentMode(self.config['agent']['mode'])
        
        # Initialize strategy-specific components
        self.initialize_strategy_components()
        
        self.event_handlers = {}
        self.setup_event_handlers()
        
    def setup_event_handlers(self):
        """Setup event handlers for inter-component communication"""
        self.event_handlers.update({
            'market_opportunity': self.handle_market_opportunity,
            'risk_threshold_breach': self.handle_risk_breach,
            'portfolio_rebalance': self.handle_portfolio_rebalance,
            'sentiment_shift': self.handle_sentiment_shift
        })

    async def handle_market_opportunity(self, data: Dict):
        """Handle market opportunity events"""
        try:
            if await self._check_security_constraints("opportunity", data.get('value', 0)):
                await self.orchestrate_strategy_execution(
                    strategy_type="market_opportunity",
                    parameters=data
                )
        except Exception as e:
            self.logger.error(f"Failed to handle market opportunity: {str(e)}")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load and merge configuration files"""
        config = {}
        config_files = ['agent_config.yaml', 'network_config.yaml', 'model_config.yaml']
        
        for file in config_files:
            path = os.path.join(config_path, file)
            with open(path, 'r') as f:
                config.update(yaml.safe_load(f))
        
        return config
    
    def _setup_logging(self) -> logging.Logger:
        """Configure comprehensive logging system"""
        logger = logging.getLogger('AgentCore')
        logger.setLevel(self.config['monitoring']['log_level'])
        
        # File handler
        file_handler = logging.FileHandler(self.config['monitoring']['log_file'])
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_cdp(self) -> CdpAgentkitWrapper:
        """Initialize CDP wrapper with wallet management"""
        try:
            wallet_path = self.config['wallet']['persistent_storage']
            
            if os.path.exists(wallet_path):
                with open(wallet_path, 'r') as f:
                    wallet_data = json.load(f)
                cdp = CdpAgentkitWrapper(cdp_wallet_data=wallet_data)
            else:
                cdp = CdpAgentkitWrapper()
                wallet_data = cdp.export_wallet()
                with open(wallet_path, 'w') as f:
                    json.dump(wallet_data, f)
            
            self.logger.info("CDP wrapper initialized successfully")
            return cdp
            
        except Exception as e:
            self.logger.error(f"CDP initialization failed: {str(e)}")
            raise
    
    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize language model with configured parameters"""
        return ChatOpenAI(
            model=self.config['model']['name'],
            temperature=self.config['model']['temperature'],
            max_tokens=self.config['model']['max_tokens'],
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def _initialize_memory(self) -> ConversationBufferMemory:
        """Initialize conversation memory system"""
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def _setup_agent(self) -> Any:
        """Configure the LangChain ReAct agent"""
        tools = self.toolkit.get_tools()
        return create_react_agent(
            llm=self.llm,
            tools=tools,
            memory=self.memory
        )
    
    def _initialize_state(self) -> AgentState:
        try:
            wallet_details = self.cdp_wrapper.get_wallet_details()
            balance = float(wallet_details['balance'])
            
            # Add historical state tracking
            self.state_history = []
            self.state_snapshot_interval = self.config['monitoring'].get('state_snapshot_interval', 3600)
            
            state = AgentState(
                wallet_address=wallet_details['address'],
                current_balance=balance,
                active_contracts=[],
                pending_transactions=[],
                last_action_timestamp=datetime.now().timestamp(),
                daily_transaction_count=0,
                total_value_locked=0.0,
                performance_metrics={
                    'win_rate': 0.0,
                    'average_profit': 0.0,
                    'sharpe_ratio': 0.0
                },
                risk_metrics={
                    'var': 0.0,
                    'max_drawdown': 0.0,
                    'current_exposure': 0.0
                }
            )
            
            # Start state tracking
            asyncio.create_task(self._track_state_history(state))
            return state
            
        except Exception as e:
            self.logger.error(f"State initialization failed: {str(e)}")
            raise

    async def _track_state_history(self, state: AgentState):
        """Track historical state for analysis"""
        while True:
            self.state_history.append(state.copy())
            if len(self.state_history) > self.config['monitoring'].get('max_state_history', 1000):
                self.state_history.pop(0)
            await asyncio.sleep(self.state_snapshot_interval)
    
    async def _check_security_constraints(self, action: str, value: float) -> bool:
        """Verify action against security constraints"""
        try:
            # Check transaction value limit
            if value > self.config['security']['max_transaction_value']:
                self.logger.warning(f"Transaction value {value} exceeds maximum allowed")
                return False
            
            # Check daily transaction limit
            if self.state.daily_transaction_count >= self.config['security']['max_daily_transactions']:
                self.logger.warning("Daily transaction limit reached")
                return False
            
            # Check minimum balance requirement
            if self.state.current_balance < self.config['wallet']['min_balance_threshold']:
                await self.request_funds()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Security check failed: {str(e)}")
            return False
    
    async def request_funds(self) -> Dict:
        """Request testnet funds if balance is low"""
        try:
            result = await self.agent.ainvoke({
                "messages": [("user", "Request testnet funds from the faucet")]
            })
            
            self.logger.info("Successfully requested testnet funds")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to request funds: {str(e)}")
            raise
    
    async def deploy_token(
        self,
        name: str,
        symbol: str,
        initial_supply: int
    ) -> Tuple[str, Dict]:
        """Deploy a new ERC20 token with security checks"""
        try:
            # Security verification
            deployment_cost = self.config['contracts']['token_factory']['min_deployment_cost']
            if not await self._check_security_constraints("deploy_token", deployment_cost):
                raise ValueError("Security constraints not met for token deployment")
            
            result = await self.agent.ainvoke({
                "messages": [("user", f"""
                Deploy an ERC20 token with:
                - Name: {name}
                - Symbol: {symbol}
                - Initial Supply: {initial_supply}
                
                Verify the deployment and return the contract address.
                """)]
            })
            
            # Update agent state
            if 'contract_address' in result:
                self.state.active_contracts.append(result['contract_address'])
                self.state.daily_transaction_count += 1
            
            self.logger.info(f"Token deployment successful: {result}")
            return result['contract_address'], result
            
        except Exception as e:
            self.logger.error(f"Token deployment failed: {str(e)}")
            raise
    
    async def create_liquidity_pool(
        self,
        token_address: str,
        eth_amount: float
    ) -> Tuple[str, Dict]:
        """Create a new liquidity pool with security checks"""
        try:
            # Security verification
            if not await self._check_security_constraints("create_liquidity", eth_amount):
                raise ValueError("Security constraints not met for liquidity creation")
            
            result = await self.agent.ainvoke({
                "messages": [("user", f"""
                Create a liquidity pool for token {token_address}
                with {eth_amount} ETH initial liquidity.
                Verify the pool creation and return the pool address.
                """)]
            })
            
            # Update agent state
            if 'pool_address' in result:
                self.state.active_contracts.append(result['pool_address'])
                self.state.daily_transaction_count += 1
                self.state.total_value_locked += eth_amount
            
            self.logger.info(f"Liquidity pool creation successful: {result}")
            return result['pool_address'], result
            
        except Exception as e:
            self.logger.error(f"Liquidity pool creation failed: {str(e)}")
            raise
    
    async def analyze_market_conditions(self) -> Dict:
        """Analyze current market conditions for decision making"""
        try:
            result = await self.agent.ainvoke({
                "messages": [("user", """
                Analyze current market conditions considering:
                1. Token performance metrics
                2. Liquidity pool metrics
                3. Gas prices and network activity
                4. Market sentiment
                
                Provide recommendations for optimal actions.
                """)]
            })
            
            self.logger.info("Market analysis completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Market analysis failed: {str(e)}")
            raise
    
    async def run_autonomous_mode(self):
        """Execute autonomous operations with enhanced strategy orchestration"""
        self.logger.info("Starting autonomous mode with enhanced orchestration")

        while self.mode == AgentMode.AUTONOMOUS:
            try:
                # Start opportunity monitoring
                await self.monitor_opportunities()

                # Analyze market conditions and portfolio state
                analysis = await self.analyze_market_conditions()
                await self.optimize_portfolio()

                # Execute recommended actions with orchestration
                if 'recommended_actions' in analysis:
                    for action in analysis['recommended_actions']:
                        await self.orchestrate_strategy_execution(
                            strategy_type=action['type'],
                            parameters=action['parameters']
                        )

                await asyncio.sleep(self.config['decision_making']['decision_interval'])

            except Exception as e:
                self.logger.error(f"Enhanced autonomous mode error: {str(e)}")
                await asyncio.sleep(60)
    
    async def process_user_command(self, command: str) -> Dict:
        """Process user commands in interactive mode"""
        try:
            result = await self.agent.ainvoke({
                "messages": [("user", command)]
            })
            
            self.logger.info(f"Processed user command: {command}")
            return result
            
        except Exception as e:
            self.logger.error(f"Command processing failed: {str(e)}")
            raise
    
    def start(self):
        """Start the agent in configured mode"""
        try:
            if self.mode == AgentMode.AUTONOMOUS:
                asyncio.run(self.run_autonomous_mode())
            elif self.mode == AgentMode.INTERACTIVE:
                print("Starting interactive mode. Type 'exit' to quit.")
                while True:
                    command = input("\nEnter command: ")
                    if command.lower() == 'exit':
                        break
                    result = asyncio.run(self.process_user_command(command))
                    print(f"\nResult: {result}")
            elif self.mode == AgentMode.ANALYSIS:
                result = asyncio.run(self.analyze_market_conditions())
                print(f"\nAnalysis Result: {result}")
                
        except Exception as e:
            self.logger.error(f"Agent execution failed: {str(e)}")
            raise
        finally:
            self.logger.info("Agent shutdown complete")
    
    def initialize_strategy_components(self):
        """Initialize strategy-specific components"""
        self.dex_monitor = CDPDEXMonitor(self.cdp_wrapper)
        self.launch_detector = LaunchDetector(self.config['strategies']['token_launch'])
        self.executor = CDPExecutor(self.config['execution'])
        self.portfolio_manager = PortfolioManager(self.config['risk'])
        self.sentiment_analyzer = SentimentAnalyzer(self.config['social'])

    async def orchestrate_strategy_execution(self, strategy_type: str, parameters: Dict) -> Dict:
        """Orchestrate the execution of a specific trading strategy"""
        try:
            # Validate strategy against risk parameters
            risk_validation = await self.portfolio_manager.validate_strategy(
                strategy_type,
                parameters
            )
            
            if not risk_validation['approved']:
                self.logger.warning(f"Strategy rejected: {risk_validation['reason']}")
                return {'status': 'rejected', 'reason': risk_validation['reason']}

            # Check market conditions
            market_conditions = await self.analyze_market_conditions()
            if not market_conditions['favorable']:
                return {'status': 'delayed', 'reason': 'Unfavorable market conditions'}

            # Execute strategy
            execution_result = await self.executor.execute_strategy(
                strategy_type=strategy_type,
                parameters=parameters,
                market_conditions=market_conditions
            )

            # Update portfolio state
            await self.portfolio_manager.update_portfolio_state(execution_result)

            return {'status': 'success', 'result': execution_result}

        except Exception as e:
            self.logger.error(f"Strategy execution failed: {str(e)}")
            raise

    async def monitor_opportunities(self):
        """Monitor various opportunities across different strategies"""
        try:
            # Start monitoring tasks concurrently
            monitoring_tasks = [
                self.dex_monitor.monitor_arbitrage_opportunities(),
                self.launch_detector.monitor_new_launches(),
                self.sentiment_analyzer.monitor_sentiment()
            ]

            # Create monitoring task group
            async with asyncio.TaskGroup() as group:
                for task in monitoring_tasks:
                    group.create_task(task)

        except Exception as e:
            self.logger.error(f"Opportunity monitoring failed: {str(e)}")
            raise

    async def optimize_portfolio(self):
        """Optimize portfolio based on current market conditions and risk parameters"""
        try:
            # Get current portfolio state
            portfolio_state = await self.portfolio_manager.get_portfolio_state()
    
            # Analyze market conditions
            market_analysis = await self.analyze_market_conditions()
    
            # Generate optimization recommendations
            optimization_actions = await self.portfolio_manager.generate_optimization_actions(
                portfolio_state,
                market_analysis
            )
    
            # Execute optimization actions
            for action in optimization_actions:
                await self.orchestrate_strategy_execution(
                    strategy_type=action['type'],
                    parameters=action['parameters']
                )
    
            return {'status': 'success', 'actions': optimization_actions}
    
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {str(e)}")
            raise
    async def monitor_performance(self):
        """Monitor agent performance and implement recovery mechanisms"""
        while True:
            try:
                # Calculate performance metrics
                current_metrics = await self._calculate_performance_metrics()
                
                # Check for performance degradation
                if self._detect_performance_degradation(current_metrics):
                    await self._implement_recovery_measures()
                
                # Update state with new metrics
                self.state.performance_metrics.update(current_metrics)
                
                await asyncio.sleep(self.config['monitoring']['performance_check_interval'])
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {str(e)}")
                await asyncio.sleep(60)

    async def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            trades = await self.portfolio_manager.get_trading_history()
            return {
                'win_rate': self._calculate_win_rate(trades),
                'average_profit': self._calculate_average_profit(trades),
                'sharpe_ratio': self._calculate_sharpe_ratio(trades),
                'max_drawdown': self._calculate_max_drawdown(trades)
            }
        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {str(e)}")
            raise
    async def emergency_shutdown(self, reason: str):
        """Implement emergency shutdown with position unwinding"""
        try:
            self.logger.warning(f"Initiating emergency shutdown: {reason}")
            
            # Stop all monitoring tasks
            for task in asyncio.all_tasks():
                if task != asyncio.current_task():
                    task.cancel()
            
            # Unwind open positions
            await self._unwind_positions()
            
            # Save final state
            await self._save_final_state()
            
            # Notify administrators
            await self._send_emergency_notification(reason)
            
        except Exception as e:
            self.logger.error(f"Emergency shutdown failed: {str(e)}")
            raise
        finally:
            sys.exit(1)