from langchain_openai import ChatOpenAI
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Notice we're not importing CdpToolkit and CdpAgentkitWrapper here directly
class CDPIntegration:
    """Handles core CDP functionality and tooling."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced CDP agent with AI components."""
        self.config = config
        
        # Initialize CDP Core Components
        self.llm = ChatOpenAI(model=config.get("llm_model", "gpt-4o-mini"))
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        self.tools = self.toolkit.get_tools()
        
    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize the language model."""
        return ChatOpenAI(
            model=self.config.get('model', 'gpt-4o-mini'),
            temperature=self.config.get('temperature', 0.7)
        )
        
    def _initialize_memory(self) -> ConversationBufferMemory:
        """Initialize conversation memory."""
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize CDP components with proper error handling."""
        try:
            # Import here to avoid circular imports
            from cdp_langchain.agent_toolkits import CdpToolkit
            from cdp_langchain.utils import CdpAgentkitWrapper
            
            self.wrapper = CdpAgentkitWrapper()
            self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.wrapper)
            self.tools = self.toolkit.get_tools()
            
            return {
                "status": "success",
                "message": "CDP components initialized successfully"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"CDP initialization failed: {str(e)}"
            }
            
    def get_components(self) -> Dict[str, Any]:
        """Get initialized CDP components."""
        return {
            "llm": self.llm,
            "memory": self.memory,
            "wrapper": self.wrapper,
            "toolkit": self.toolkit,
            "tools": self.tools
        }

# Create singleton instance
cdp_instance = CDPIntegration({})

# Export only what's needed
__all__ = ['cdp_instance', 'CDPIntegration']