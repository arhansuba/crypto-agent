from swarm import Swarm
from typing import Dict, Any, List
import pytest
import asyncio
from agents import AIEnhancedAgent
from utils.logger import get_logger
from utils.config_manager import ConfigManager

logger = get_logger(__name__)
client = Swarm()

class AgentEvaluator:
    """Comprehensive evaluation system for AI-enhanced CDP agent."""
    
    def __init__(self, config_path: str):
        self.config = ConfigManager(config_path)
        self.agent = AIEnhancedAgent(self.config.get_agent_config())

    async def run_and_get_tool_calls(self, query: str) -> List[Dict[str, Any]]:
        """Execute agent query and return tool calls with performance metrics."""
        try:
            message = {"role": "user", "content": query}
            start_time = asyncio.get_event_loop().time()
            
            response = await client.run(
                agent=self.agent,
                messages=[message],
                execute_tools=False
            )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Record performance metrics
            await self._record_execution_metrics(
                query_type="tool_call",
                execution_time=execution_time,
                response=response
            )
            
            return response.messages[-1].get("tool_calls")
            
        except Exception as e:
            logger.error(f"Tool call execution failed: {str(e)}")
            return []

    async def evaluate_trading_strategy(self, strategy_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate AI trading strategy performance."""
        try:
            # Execute strategy in test mode
            result = await self.agent.execute_ai_trading(
                strategy_type=strategy_type,
                params=params,
                test_mode=True
            )
            
            # Analyze performance
            performance_metrics = await self._analyze_strategy_performance(result)
            
            return {
                "strategy_type": strategy_type,
                "execution_result": result,
                "performance_metrics": performance_metrics,
                "recommendations": await self._generate_optimization_recommendations(
                    performance_metrics
                )
            }
            
        except Exception as e:
            logger.error(f"Strategy evaluation failed: {str(e)}")
            return {"status": "error", "reason": str(e)}

    async def _record_execution_metrics(self, query_type: str, execution_time: float, response: Any):
        """Record execution metrics for performance analysis."""
        try:
            metrics = {
                "query_type": query_type,
                "execution_time": execution_time,
                "response_status": "success" if response else "failed",
                "timestamp": asyncio.get_event_loop().time()
            }
            
            await self.agent.record_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Failed to record metrics: {str(e)}")

# Test Cases
@pytest.mark.asyncio
async def test_token_creation():
    """Test token creation with market analysis."""
    evaluator = AgentEvaluator("config/test_config.yaml")
    
    result = await evaluator.run_and_get_tool_calls(
        "Create a new token called 'TestToken' with symbol 'TEST'"
    )
    
    assert len(result) == 1
    assert result[0]["function"]["name"] == "create_token_with_analysis"

@pytest.mark.asyncio
async def test_trade_execution():
    """Test trade execution with risk management."""
    evaluator = AgentEvaluator("config/test_config.yaml")
    
    result = await evaluator.run_and_get_tool_calls(
        "Execute a trade of 0.1 ETH to USDC with optimal timing"
    )
    
    assert len(result) == 1
    assert result[0]["function"]["name"] == "execute_ai_trading"

@pytest.mark.asyncio
async def test_market_analysis():
    """Test market analysis capabilities."""
    evaluator = AgentEvaluator("config/test_config.yaml")
    
    result = await evaluator.run_and_get_tool_calls(
        "Analyze current market conditions for token deployment"
    )
    
    assert len(result) == 1
    assert result[0]["function"]["name"] == "analyze_market_conditions"

@pytest.mark.parametrize(
    "query",
    [
        "What's the current gas price?",
        "Should I deploy a token now?",
        "Is it a good time to provide liquidity?",
    ]
)
async def test_ai_analysis_queries(query):
    """Test AI analysis response to various queries."""
    evaluator = AgentEvaluator("config/test_config.yaml")
    result = await evaluator.run_and_get_tool_calls(query)
    assert result is not None

@pytest.mark.asyncio
async def test_risk_management():
    """Test risk management system."""
    evaluator = AgentEvaluator("config/test_config.yaml")
    
    strategy_evaluation = await evaluator.evaluate_trading_strategy(
        strategy_type="arbitrage",
        params={
            "amount": 1.0,
            "token_pair": ["ETH", "USDC"],
            "max_slippage": 0.01
        }
    )
    
    assert strategy_evaluation["status"] != "error"
    assert "risk_assessment" in strategy_evaluation["performance_metrics"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])