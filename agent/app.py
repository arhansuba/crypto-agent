from fastapi import FastAPI, WebSocket, Request, HTTPException, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import APIKeyHeader
import asyncio
from typing import Optional, Dict, Any, List
import uvicorn

from agents import AIEnhancedAgent
from utils.logger import get_logger
from utils.config_manager import ConfigManager
from risk.portfolio_manager import PortfolioManager
from analysis.market_analyzer import MarketAnalyzer
from analysis.sentiment_scanner import SentimentScanner
# Add CDP-specific imports
from cdp_core.base_agent import BaseCDPAgent
from strategies.arbitrage.cdp_dex_monitor import CDPDEXMonitor
from execution.cdp_executor import CDPExecutor

logger = get_logger(__name__)

class AITradingApplication:
    def __init__(self):
        self.app = FastAPI(title="AI-Enhanced CDP Trading Platform")
        self.config = ConfigManager("config/agent_config.yaml")
        
        # Initialize existing components
        self.agent = AIEnhancedAgent(self.config.get_agent_config())
        self.market_analyzer = MarketAnalyzer(self.config.get_market_config())
        self.portfolio_manager = PortfolioManager(self.config.get_risk_config())
        self.sentiment_scanner = SentimentScanner(self.config.get_sentiment_config())
        
        # Initialize CDP components
        self.cdp_agent = BaseCDPAgent(self.config.get_agent_config())
        self.dex_monitor = CDPDEXMonitor(self.cdp_agent)
        self.executor = CDPExecutor()
        
        self.active_connections: List[WebSocket] = []
        self.agent_tasks: Dict[str, asyncio.Task] = {}
        self.strategy_monitors: Dict[str, asyncio.Task] = {}
        
        self._setup_routes()
        self._setup_static_files()
        self._setup_security()

    def _setup_routes(self):
        """Initialize FastAPI routes and WebSocket endpoints."""
        # Keep existing routes
        self.app.get("/")(self.root)
        self.app.websocket("/ws")(self.websocket_endpoint)
        self.app.post("/api/execute-trade")(self.execute_trade)
        self.app.get("/api/market-analysis")(self.get_market_analysis)
        self.app.get("/api/portfolio-status")(self.get_portfolio_status)
        
        # Add new CDP-specific routes
        self.app.post("/api/deploy-token")(self.deploy_token)
        self.app.post("/api/create-liquidity-pool")(self.create_liquidity_pool)
        self.app.get("/api/monitor-arbitrage")(self.monitor_arbitrage)

    def _setup_security(self):
        """Configure API security and rate limiting."""
        self.api_key_header = APIKeyHeader(name="X-API-Key")
        
        @self.app.middleware("http")
        async def validate_api_key(request: Request, call_next):
            if request.url.path.startswith("/api"):
                api_key = request.headers.get("X-API-Key")
                if not await self._verify_api_key(api_key):
                    raise HTTPException(status_code=401, detail="Invalid API key")
            return await call_next(request)

    async def deploy_token(self, token_params: Dict[str, Any], background_tasks: BackgroundTasks):
        """Deploy a new token with CDP integration."""
        try:
            deployment_result = await self.cdp_agent.deploy_token(
                name=token_params["name"],
                symbol=token_params["symbol"],
                initial_supply=token_params["initial_supply"]
            )
            
            # Monitor the new token in background
            background_tasks.add_task(
                self.dex_monitor.monitor_token,
                deployment_result["contract_address"]
            )
            
            return {"status": "success", "result": deployment_result}
            
        except Exception as e:
            logger.error(f"Token deployment error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def create_liquidity_pool(self, pool_params: Dict[str, Any]):
        """Create a new liquidity pool with CDP integration."""
        try:
            pool_result = await self.cdp_agent.create_liquidity_pool(
                token_address=pool_params["token_address"],
                eth_amount=pool_params["eth_amount"]
            )
            
            # Update portfolio tracking
            await self.portfolio_manager.track_liquidity_position(pool_result)
            
            return {"status": "success", "result": pool_result}
            
        except Exception as e:
            logger.error(f"Liquidity pool creation error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def monitor_arbitrage(self, websocket: WebSocket):
        """Monitor and execute arbitrage opportunities."""
        try:
            while True:
                opportunities = await self.dex_monitor.scan_opportunities()
                
                for opportunity in opportunities:
                    if await self.portfolio_manager.validate_arbitrage(opportunity):
                        execution_result = await self.executor.execute_arbitrage(
                            opportunity,
                            self.cdp_agent
                        )
                        await websocket.send_json({
                            "type": "arbitrage_execution",
                            "result": execution_result
                        })
                
                await asyncio.sleep(self.config.get("arbitrage_scan_interval"))
                
        except Exception as e:
            logger.error(f"Arbitrage monitoring error: {str(e)}")
            await self._send_error(websocket, "Arbitrage monitoring failed", str(e))

    async def agent_monitoring_loop(self, websocket: WebSocket):
        """Enhanced monitoring loop with CDP integration."""
        try:
            while True:
                # Get CDP-specific metrics
                cdp_metrics = await self.cdp_agent.get_metrics()
                
                # Combine with existing analysis
                market_analysis = await self.market_analyzer.analyze_market_conditions()
                sentiment_data = await self.sentiment_scanner.analyze_market_sentiment()
                
                combined_analysis = {
                    **market_analysis,
                    "cdp_metrics": cdp_metrics,
                    "sentiment": sentiment_data
                }

                if await self.portfolio_manager.should_execute_trades(combined_analysis):
                    strategy = await self.agent.generate_trading_strategy(
                        market_analysis=combined_analysis,
                        sentiment_data=sentiment_data
                    )
                    
                    execution_result = await self.executor.execute_strategy(
                        strategy,
                        self.cdp_agent
                    )

                    await self._send_execution_update(websocket, execution_result)

                await asyncio.sleep(self.config.get("monitoring_interval"))

        except Exception as e:
            logger.error(f"Enhanced monitoring error: {str(e)}")
            await self._send_error(websocket, "Monitoring failed", str(e))
    async def root(self, request: Request):
        """Serve the main application page."""
        return self.templates.TemplateResponse("index.html", {
            "request": request,
            "agent_status": await self.agent.get_status()
        })

    async def websocket_endpoint(self, websocket: WebSocket):
        """Handle WebSocket connections for real-time updates."""
        await websocket.accept()
        self.active_connections.append(websocket)

        try:
            await self.display_agent_status(websocket)
            
            while True:
                data = await websocket.receive_json()
                await self._handle_websocket_message(data, websocket)

        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
        finally:
            await self._cleanup_connection(websocket)

    async def execute_trade(self, trade_request: Dict[str, Any]):
        """Execute trade with AI optimization."""
        try:
            execution_result = await self.agent.execute_ai_trading(
                strategy_type=trade_request["strategy_type"],
                params=trade_request["params"]
            )
            return {"status": "success", "result": execution_result}
        except Exception as e:
            logger.error(f"Trade execution error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def run(self):
        """Start the FastAPI application."""
        uvicorn.run(self.app, host="0.0.0.0", port=3000)

if __name__ == "__main__":
    trading_app = AITradingApplication()
    trading_app.run()