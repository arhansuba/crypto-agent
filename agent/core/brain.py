"""
AI decision-making logic for the crypto agent.
Handles market analysis, risk assessment, and strategic decision-making.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
import json
import logging

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

@dataclass
class MarketCondition:
    """Represents current market conditions and metrics"""
    timestamp: float
    gas_price: float
    network_load: float
    sentiment_score: float
    liquidity_metrics: Dict[str, float]
    token_metrics: Dict[str, Dict[str, float]]
    risk_indicators: Dict[str, float]

@dataclass
class ActionRecommendation:
    """Represents a recommended action with associated parameters"""
    action_type: str
    confidence_score: float
    parameters: Dict[str, any]
    expected_outcome: Dict[str, float]
    risk_assessment: Dict[str, float]

class AgentBrain:
    """
    Advanced decision-making system for the AI crypto agent.
    Handles market analysis, risk assessment, and action planning.
    """
    
    def __init__(self, config: Dict, llm: ChatOpenAI, logger: logging.Logger):
        """Initialize the decision-making system"""
        self.config = config
        self.llm = llm
        self.logger = logger
        
        # Initialize strategic components
        self.market_analyzer = self._setup_market_analyzer()
        self.risk_manager = self._setup_risk_manager()
        self.decision_engine = self._setup_decision_engine()
        
        # Load historical data
        self.market_history = []
        self.action_history = []
        
    def _setup_market_analyzer(self) -> LLMChain:
        """Configure market analysis chain"""
        market_template = """
        Analyze the current market conditions with the following data:
        Gas Price: {gas_price} gwei
        Network Load: {network_load}%
        Sentiment Score: {sentiment_score}
        Liquidity Metrics: {liquidity_metrics}
        Token Performance: {token_metrics}
        
        Provide a detailed analysis considering:
        1. Market trends and patterns
        2. Network efficiency indicators
        3. Token performance metrics
        4. Liquidity depth and stability
        5. Risk indicators
        
        Return a structured analysis with specific metrics and recommendations.
        """
        
        return LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=market_template,
                input_variables=[
                    "gas_price", "network_load", "sentiment_score",
                    "liquidity_metrics", "token_metrics"
                ]
            )
        )
    
    def _setup_risk_manager(self) -> LLMChain:
        """Configure risk assessment chain"""
        risk_template = """
        Evaluate the risks for the proposed action:
        Action Type: {action_type}
        Parameters: {parameters}
        Current Market Conditions: {market_conditions}
        Historical Performance: {historical_metrics}
        
        Consider the following risk factors:
        1. Market volatility and exposure
        2. Transaction costs and gas prices
        3. Smart contract risks
        4. Liquidity risks
        5. Regulatory considerations
        
        Provide a comprehensive risk assessment with specific metrics and mitigation strategies.
        """
        
        return LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=risk_template,
                input_variables=[
                    "action_type", "parameters", "market_conditions",
                    "historical_metrics"
                ]
            )
        )
    
    def _setup_decision_engine(self) -> LLMChain:
        """Configure strategic decision-making chain"""
        decision_template = """
        Make a strategic decision based on:
        Market Analysis: {market_analysis}
        Risk Assessment: {risk_assessment}
        Available Actions: {available_actions}
        Portfolio State: {portfolio_state}
        Historical Performance: {historical_performance}
        
        Consider the strategy parameters:
        - Risk Tolerance: {risk_tolerance}
        - Performance Targets: {performance_targets}
        - Resource Constraints: {resource_constraints}
        
        Provide a detailed decision with:
        1. Recommended actions and timing
        2. Expected outcomes and metrics
        3. Risk mitigation strategies
        4. Performance monitoring indicators
        """
        
        return LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=decision_template,
                input_variables=[
                    "market_analysis", "risk_assessment", "available_actions",
                    "portfolio_state", "historical_performance", "risk_tolerance",
                    "performance_targets", "resource_constraints"
                ]
            )
        )
    
    async def analyze_market_conditions(self) -> MarketCondition:
        """Perform comprehensive market analysis"""
        try:
            # Gather current market data
            current_data = await self._gather_market_data()
            
            # Analyze using market analyzer chain
            analysis = await self.market_analyzer.arun(
                gas_price=current_data["gas_price"],
                network_load=current_data["network_load"],
                sentiment_score=current_data["sentiment_score"],
                liquidity_metrics=json.dumps(current_data["liquidity_metrics"]),
                token_metrics=json.dumps(current_data["token_metrics"])
            )
            
            # Create market condition object
            market_condition = MarketCondition(
                timestamp=datetime.now().timestamp(),
                gas_price=current_data["gas_price"],
                network_load=current_data["network_load"],
                sentiment_score=float(analysis["sentiment_score"]),
                liquidity_metrics=analysis["liquidity_metrics"],
                token_metrics=analysis["token_metrics"],
                risk_indicators=analysis["risk_indicators"]
            )
            
            # Update historical data
            self.market_history.append(market_condition)
            if len(self.market_history) > self.config["memory"]["max_history_size"]:
                self.market_history.pop(0)
            
            return market_condition
            
        except Exception as e:
            self.logger.error(f"Market analysis failed: {str(e)}")
            raise
    
    async def assess_action_risk(
        self,
        action_type: str,
        parameters: Dict
    ) -> Dict[str, float]:
        """Evaluate risks for a proposed action"""
        try:
            # Get current market conditions
            market_condition = await self.analyze_market_conditions()
            
            # Get historical metrics
            historical_metrics = self._calculate_historical_metrics()
            
            # Perform risk assessment
            risk_assessment = await self.risk_manager.arun(
                action_type=action_type,
                parameters=json.dumps(parameters),
                market_conditions=json.dumps(market_condition.__dict__),
                historical_metrics=json.dumps(historical_metrics)
            )
            
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {str(e)}")
            raise
    
    async def make_decision(
        self,
        available_actions: List[str],
        portfolio_state: Dict
    ) -> ActionRecommendation:
        """Make strategic decision based on current conditions"""
        try:
            # Get current market analysis
            market_condition = await self.analyze_market_conditions()
            
            # Assess risks for each available action
            action_risks = {}
            for action in available_actions:
                risk = await self.assess_action_risk(
                    action,
                    self.config["action_parameters"][action]
                )
                action_risks[action] = risk
            
            # Get historical performance metrics
            historical_performance = self._calculate_historical_metrics()
            
            # Make strategic decision
            decision = await self.decision_engine.arun(
                market_analysis=json.dumps(market_condition.__dict__),
                risk_assessment=json.dumps(action_risks),
                available_actions=json.dumps(available_actions),
                portfolio_state=json.dumps(portfolio_state),
                historical_performance=json.dumps(historical_performance),
                risk_tolerance=self.config["decision_making"]["risk_tolerance"],
                performance_targets=json.dumps(
                    self.config["decision_making"]["performance_targets"]
                ),
                resource_constraints=json.dumps(
                    self.config["decision_making"]["resource_constraints"]
                )
            )
            
            # Create recommendation object
            recommendation = ActionRecommendation(
                action_type=decision["selected_action"],
                confidence_score=float(decision["confidence"]),
                parameters=decision["parameters"],
                expected_outcome=decision["expected_outcome"],
                risk_assessment=decision["risk_assessment"]
            )
            
            # Update action history
            self.action_history.append({
                "timestamp": datetime.now().timestamp(),
                "recommendation": recommendation.__dict__,
                "market_condition": market_condition.__dict__
            })
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Decision making failed: {str(e)}")
            raise
    
    def _calculate_historical_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics from historical data"""
        try:
            if not self.market_history:
                return {}
            
            metrics = {
                "average_gas_price": np.mean([m.gas_price for m in self.market_history]),
                "average_network_load": np.mean([m.network_load for m in self.market_history]),
                "sentiment_trend": np.mean([m.sentiment_score for m in self.market_history]),
                "volatility": np.std([m.sentiment_score for m in self.market_history]),
                "success_rate": self._calculate_success_rate(),
                "risk_adjusted_return": self._calculate_risk_adjusted_return()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Metric calculation failed: {str(e)}")
            return {}
    
    def _calculate_success_rate(self) -> float:
        """Calculate the success rate of past decisions"""
        if not self.action_history:
            return 0.0
            
        successful_actions = sum(
            1 for action in self.action_history
            if action["recommendation"]["expected_outcome"]["success"]
        )
        return successful_actions / len(self.action_history)
    
    def _calculate_risk_adjusted_return(self) -> float:
        """Calculate risk-adjusted return metrics"""
        if not self.action_history:
            return 0.0
            
        returns = [
            action["recommendation"]["expected_outcome"]["return"]
            for action in self.action_history
        ]
        
        if not returns:
            return 0.0
            
        return np.mean(returns) / (np.std(returns) + 1e-6)
    
    async def _gather_market_data(self) -> Dict:
        """Gather current market data from various sources"""
        try:
            # Implement actual data gathering logic here
            return {
                "gas_price": await self._get_gas_price(),
                "network_load": await self._get_network_load(),
                "sentiment_score": await self._get_sentiment_score(),
                "liquidity_metrics": await self._get_liquidity_metrics(),
                "token_metrics": await self._get_token_metrics()
            }
        except Exception as e:
            self.logger.error(f"Data gathering failed: {str(e)}")
            raise