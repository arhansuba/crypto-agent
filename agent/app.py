from fastapi import FastAPI, WebSocket, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
from agents import based_agent, agent_wallet, request_eth_from_faucet, get_balance
from swarm import Swarm
import uvicorn
from typing import Optional, Dict, Any, List

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Store active WebSocket connections and agent task
active_connections: List[WebSocket] = []
agent_task: Optional[asyncio.Task] = None
client = Swarm()

async def display_initial_wallet_info(websocket: WebSocket):
    """Display initial wallet information via WebSocket."""
    try:
        # Display wallet address
        await websocket.send_json({
            "message": f"Agent wallet address: {agent_wallet.default_address.address_id}",
            "type": "wallet"
        })

        # Request ETH from faucet and display transaction
        try:
            faucet_result = request_eth_from_faucet()
            await websocket.send_json({
                "message": f"Faucet transaction: {faucet_result}",
                "type": "transaction"
            })
        except Exception as e:
            await websocket.send_json({
                "message": f"Error requesting from faucet: {str(e)}",
                "type": "system"
            })

        # Display current balance
        try:
            balance = get_balance("eth")
            await websocket.send_json({
                "message": f"Current ETH balance: {balance}",
                "type": "wallet"
            })
        except Exception as e:
            await websocket.send_json({
                "message": f"Error getting balance: {str(e)}",
                "type": "system"
            })

    except Exception as e:
        await websocket.send_json({
            "message": f"Error displaying wallet info: {str(e)}",
            "type": "system"
        })

async def process_streaming_response(response, websocket: WebSocket):
    """Process streaming responses from the agent and send them via WebSocket."""
    content = ""
    last_sender = ""

    try:
        for chunk in response:
            if "sender" in chunk:
                last_sender = chunk["sender"]

            if "content" in chunk and chunk["content"] is not None:
                if not content and last_sender:
                    await websocket.send_json({
                        "message": chunk["content"],
                        "type": "agent"
                    })
                content += chunk["content"]

            if "tool_calls" in chunk and chunk["tool_calls"] is not None:
                for tool_call in chunk["tool_calls"]:
                    f = tool_call["function"]
                    name = f["name"]
                    if name:
                        await websocket.send_json({
                            "message": f"Executing: {name}()",
                            "type": "tool"
                        })

            if "delim" in chunk and chunk["delim"] == "end" and content:
                content = ""

            if "response" in chunk:
                return chunk["response"]

    except Exception as e:
        await websocket.send_json({
            "message": f"Error processing response: {str(e)}",
            "type": "system"
        })

async def agent_loop(websocket: WebSocket):
    """Main agent loop that runs autonomously."""
    messages = []
    interval = 10  # seconds between actions

    try:
        while True:
            try:
                # Generate thought
                thought = (
                    f"It's been {interval} seconds. I want you to do some sort of "
                    "onchain action based on my capabilities. Let's get crazy and "
                    "creative! Don't take any more input from me, and if this is "
                    "your first command don't generate art"
                )
                messages.append({"role": "user", "content": thought})

                await websocket.send_json({
                    "message": thought,
                    "type": "system"
                })

                # Run the agent
                response = client.run(
                    agent=based_agent,
                    messages=messages,
                    stream=True
                )

                # Process the streaming response
                await process_streaming_response(response, websocket)

                # Wait for the specified interval
                await asyncio.sleep(interval)

            except Exception as e:
                await websocket.send_json({
                    "message": f"Error in agent loop: {str(e)}",
                    "type": "system"
                })
                await asyncio.sleep(interval)

    except Exception as e:
        await websocket.send_json({
            "message": f"Critical error: {str(e)}. Please refresh the page.",
            "type": "system"
        })

@app.get("/")
async def root(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections and agent control."""
    global agent_task

    await websocket.accept()
    active_connections.append(websocket)

    try:
        await websocket.send_json({
            "message": "Connected to server. Click \"Start Agent\" to begin...",
            "type": "system"
        })

        while True:
            data = await websocket.receive_json()

            if data.get("action") == "start":
                if agent_task is None or agent_task.done():
                    await websocket.send_json({
                        "message": "Starting agent...",
                        "type": "system"
                    })
                    # Display wallet info before starting the agent
                    await display_initial_wallet_info(websocket)
                    agent_task = asyncio.create_task(agent_loop(websocket))

            elif data.get("action") == "stop":
                if agent_task and not agent_task.done():
                    agent_task.cancel()
                    await websocket.send_json({
                        "message": "Agent stopped.",
                        "type": "system"
                    })

    except Exception as e:
        await websocket.send_json({
            "message": f"WebSocket error: {str(e)}",
            "type": "system"
        })

    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)