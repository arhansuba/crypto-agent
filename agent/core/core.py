"""
Core implementation of the AI Crypto Agent using CDP Agentkit and LangChain.
This module handles the primary agent logic, decision-making, and blockchain interactions.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import yaml

from agent.core.memory import AgentState
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langgraph.prebuilt import create_react_agent

class AgentMode(Enum):
    # Add your enum values here
    pass

# Add your dataclass and other code here

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
        """Initialize agent state with current blockchain data"""
        try:
            wallet_details = self.cdp_wrapper.get_wallet_details()
            balance = float(wallet_details['balance'])
            
            return AgentState(
                wallet_address=wallet_details['address'],
                current_balance=balance,
                active_contracts=[],
                pending_transactions=[],
                last_action_timestamp=datetime.now().timestamp(),
                daily_transaction_count=0,
                total_value_locked=0.0
            )
            
        except Exception as e:
            self.logger.error(f"State initialization failed: {str(e)}")
            raise
    
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
        """Execute autonomous operations based on market analysis"""
        self.logger.info("Starting autonomous mode")
        
        while self.mode == AgentMode.AUTONOMOUS:
            try:
                # Analyze market conditions
                analysis = await self.analyze_market_conditions()
                
                # Execute recommended actions
                if 'recommended_actions' in analysis:
                    for action in analysis['recommended_actions']:
                        if action['type'] == 'deploy_token':
                            await self.deploy_token(
                                action['name'],
                                action['symbol'],
                                action['initial_supply']
                            )
                        elif action['type'] == 'create_liquidity':
                            await self.create_liquidity_pool(
                                action['token_address'],
                                action['eth_amount']
                            )
                
                # Wait for next iteration
                await asyncio.sleep(self.config['decision_making']['decision_interval'])
                
            except Exception as e:
                self.logger.error(f"Autonomous mode error: {str(e)}")
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