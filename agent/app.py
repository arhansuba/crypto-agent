from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
from typing import Optional, Dict, Any, List
import uvicorn

from agents import AIEnhancedAgent
from utils.logger import get_logger
from utils.config_manager import ConfigManager
from risk.portfolio_manager import PortfolioManager
from analysis.market_analyzer import MarketAnalyzer
from analysis.sentiment_scanner import SentimentScanner

logger = get_logger(__name__)

class AITradingApplication:
    def __init__(self):
        self.app = FastAPI(title="AI-Enhanced CDP Trading Platform")
        self.config = ConfigManager("config/agent_config.yaml")
        self.agent = AIEnhancedAgent(self.config.get_agent_config())
        self.market_analyzer = MarketAnalyzer(self.config.get_market_config())
        self.portfolio_manager = PortfolioManager(self.config.get_risk_config())
        self.sentiment_scanner = SentimentScanner(self.config.get_sentiment_config())
        
        self.active_connections: List[WebSocket] = []
        self.agent_tasks: Dict[str, asyncio.Task] = {}
        
        self._setup_routes()
        self._setup_static_files()

    def _setup_routes(self):
        """Initialize FastAPI routes and WebSocket endpoints."""
        self.app.get("/")(self.root)
        self.app.websocket("/ws")(self.websocket_endpoint)
        self.app.post("/api/execute-trade")(self.execute_trade)
        self.app.get("/api/market-analysis")(self.get_market_analysis)
        self.app.get("/api/portfolio-status")(self.get_portfolio_status)

    def _setup_static_files(self):
        """Configure static files and templates."""
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        self.templates = Jinja2Templates(directory="templates")

    async def display_agent_status(self, websocket: WebSocket):
        """Send agent status information via WebSocket."""
        try:
            wallet_info = await self.agent.get_wallet_info()
            market_state = await self.market_analyzer.get_current_state()
            portfolio_status = await self.portfolio_manager.get_status()

            await websocket.send_json({
                "wallet_info": wallet_info,
                "market_state": market_state,
                "portfolio_status": portfolio_status,
                "type": "status_update"
            })

        except Exception as e:
            logger.error(f"Status display error: {str(e)}")
            await self._send_error(websocket, "Status update failed", str(e))

    async def process_agent_response(self, response: Dict[str, Any], websocket: WebSocket):
        """Process and send agent responses via WebSocket."""
        try:
            for chunk in response:
                if chunk.get("type") == "market_update":
                    await self._handle_market_update(chunk, websocket)
                elif chunk.get("type") == "trade_execution":
                    await self._handle_trade_execution(chunk, websocket)
                elif chunk.get("type") == "risk_alert":
                    await self._handle_risk_alert(chunk, websocket)

        except Exception as e:
            logger.error(f"Response processing error: {str(e)}")
            await self._send_error(websocket, "Response processing failed", str(e))

    async def agent_monitoring_loop(self, websocket: WebSocket):
        """Main monitoring loop for the AI trading agent."""
        try:
            while True:
                # Analyze market conditions
                market_analysis = await self.market_analyzer.analyze_market_conditions()
                sentiment_data = await self.sentiment_scanner.analyze_market_sentiment()

                # Generate and execute trading strategy
                if await self.portfolio_manager.should_execute_trades(market_analysis):
                    strategy = await self.agent.generate_trading_strategy(
                        market_analysis=market_analysis,
                        sentiment_data=sentiment_data
                    )
                    
                    execution_result = await self.agent.execute_ai_trading(
                        strategy_type=strategy["type"],
                        params=strategy["params"]
                    )

                    await self._send_execution_update(websocket, execution_result)

                await asyncio.sleep(self.config.get("monitoring_interval"))

        except Exception as e:
            logger.error(f"Agent monitoring error: {str(e)}")
            await self._send_error(websocket, "Agent monitoring failed", str(e))

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