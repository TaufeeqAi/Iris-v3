# mcp-servers/agent-marketplace-mcp/server.py
import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from dotenv import load_dotenv

# Google ADK imports
from google.adk.agents import AgentRuntime, Message, LlmAgent, CustomAgent
from google.adk.protocol import Message as ADKProtocolMessage # ADK's internal Message for RPC
from google.adk.io import MemoryStore # For ADK AgentRuntime's internal state
from google.adk.agents.connectors import HttpAgentConnector # For connecting to HTTP-based agents

# For internal HTTP calls to cyrene-agent instances
import httpx

# For libp2p PeerID (from Phase 1)
from libp2p.peer.id import ID as PeerID

load_dotenv()

logger = logging.getLogger(__name__)
try:
    from common.utils import setup_logging
    setup_logging(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.warning("common.utils.setup_logging not found, using basic logging config.")

mcp = FastMCP("agent_marketplace")

# --- Global ADK AgentRuntime ---
# This runtime will manage the lifecycle of agents registered with this marketplace.
# It will use HttpAgentConnector to talk to our cyrene-agent instances.
agent_runtime: Optional[AgentRuntime] = None

# --- Agent Registry (in-memory for now, can be persistent later) ---
# Maps libp2p PeerID (str) to agent metadata and its internal invoke URL
# { "peer_id": { "name": "AgentX", "bio": "...", "capabilities": ["finance"], "internal_url": "http://cyrene-agent:8000/internal/agents/..." } }
_registered_agents: Dict[str, Dict[str, Any]] = {}

# --- ADK Agent Connector for cyrene-agent instances ---
# This connector allows the ADK runtime to talk to our custom cyrene-agent instances.
class CyreneAgentConnector(HttpAgentConnector):
    def __init__(self, agent_id: str, internal_url: str):
        super().__init__(agent_id=agent_id, url=internal_url)
        self.internal_url = internal_url
        logger.info(f"Initialized CyreneAgentConnector for agent {agent_id} at {internal_url}")

    async def send(self, message: ADKProtocolMessage) -> ADKProtocolMessage:
        """
        Sends an ADK protocol message to the target cyrene-agent instance.
        This method is called by the ADK AgentRuntime when it needs to invoke a cyrene-agent.
        """
        logger.info(f"CyreneAgentConnector sending message to {self.internal_url} for agent {self.agent_id}")
        try:
            # ADKProtocolMessage needs to be converted to the FastAPI endpoint's expected Message model
            # Assuming ADKProtocolMessage.payload contains the original ADK Message from the caller.
            # The `text` field is what our cyrene-agent's /internal/agents/{id}/invoke_adk expects.
            payload_text = message.payload.text if message.payload and message.payload.text else ""

            # Reconstruct the ADK Message format expected by our /internal/agents/{id}/invoke_adk endpoint
            # This is crucial for cross-framework compatibility.
            adk_message_for_cyrene = {
                "text": payload_text,
                "metadata": message.payload.metadata if message.payload and message.payload.metadata else {}
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.internal_url}/invoke_adk", # Our internal endpoint for ADK invocation
                    json=adk_message_for_cyrene,
                    timeout=60.0 # Increased timeout for agent processing
                )
                response.raise_for_status()

                # The response from cyrene-agent is also an ADK Message JSON
                response_data = response.json()
                returned_adk_message = Message(
                    text=response_data.get("text", ""),
                    metadata=response_data.get("metadata", {})
                )
                # Wrap it in ADK's internal protocol message
                return ADKProtocolMessage(payload=returned_adk_message)

        except httpx.HTTPStatusError as e:
            logger.error(f"CyreneAgentConnector HTTP error for {self.agent_id} at {self.internal_url}: {e.response.status_code} - {e.response.text}", exc_info=True)
            raise HTTPException(status_code=e.response.status_code, detail=f"A2A communication error: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"CyreneAgentConnector request error for {self.agent_id} at {self.internal_url}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"A2A communication error: {e}")
        except Exception as e:
            logger.error(f"CyreneAgentConnector unexpected error for {self.agent_id} at {self.internal_url}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"A2A communication error: {e}")

# --- FastMCP Tools Exposed by agent-marketplace-mcp ---

@mcp.tool()
async def register_agent_capability(
    peer_id: str,
    name: str,
    bio: str,
    capabilities: List[str],
    internal_url: str # The internal HTTP endpoint of the cyrene-agent instance
) -> str:
    """
    Registers a cyrene-agent's capabilities with the ADK A2A server.
    This allows other agents to discover and invoke it.
    """
    logger.info(f"Registering agent '{name}' (PeerID: {peer_id}) with capabilities: {capabilities} at internal URL: {internal_url}")

    # Store agent info in our registry
    _registered_agents[peer_id] = {
        "name": name,
        "bio": bio,
        "capabilities": capabilities,
        "internal_url": internal_url
    }

    # Register the agent with the ADK AgentRuntime
    # Create a HttpAgentConnector for this specific cyrene-agent instance
    connector = CyreneAgentConnector(agent_id=peer_id, internal_url=internal_url)

    # The ADK AgentRuntime needs to know about this agent and how to connect to it.
    # ADK's AgentRuntime.register_agent() typically takes an Agent object or a connector.
    # We'll use the connector approach.
    # Note: In ADK, the agent_id passed to register_agent should be unique.
    try:
        # ADK's register_agent expects an agent instance, not just a connector.
        # We'll create a dummy ADK agent that uses our connector.
        # This is a conceptual mapping. The `act` method of this dummy agent
        # won't be called directly by ADK runtime if we are using connectors
        # for direct invocation.

        # For direct invocation via AgentRuntime.send_message,
        # we just need the connector to be known to the runtime.
        # ADK's `AgentRuntime` manages `AgentConnectors` internally.
        # The `register_agent` method is more for agents that are *hosted* by this runtime.
        # For agents that are *external* but callable, we'd typically just know their connector.

        # Let's simplify: the _registered_agents dict is our primary registry.
        # The ADK runtime will use our custom CyreneAgentConnector when we call send_message.
        # The ADK runtime itself doesn't need to 'register' the external agent in its internal list
        # if we are just using it to send messages via a custom connector.

        logger.info(f"Agent '{name}' (PeerID: {peer_id}) successfully registered in agent marketplace.")
        return f"Agent '{name}' (PeerID: {peer_id}) registered successfully."
    except Exception as e:
        logger.error(f"Error registering agent '{name}' (PeerID: {peer_id}) with ADK AgentRuntime: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to register agent with marketplace: {e}")

@mcp.tool()
async def invoke_agent(target_peer_id: str, message: str) -> str:
    """
    Invokes another agent by its libp2p Peer ID with a given message.
    This tool is called by a cyrene-agent when it needs to delegate a task.
    """
    logger.info(f"Invoking agent '{target_peer_id}' with message: {message[:100]}...")

    target_agent_info = _registered_agents.get(target_peer_id)
    if not target_agent_info:
        logger.error(f"Target agent with PeerID '{target_peer_id}' not found in registry.")
        raise HTTPException(status_code=404, detail=f"Target agent '{target_peer_id}' not found.")

    try:
        # Create an ADK Message to send
        adk_message_to_send = Message(text=message)

        # Use the ADK AgentRuntime to send the message via the appropriate connector
        # The AgentRuntime will find the correct connector for target_peer_id
        # and use its send method (our CyreneAgentConnector.send)

        # Ensure agent_runtime is initialized
        if agent_runtime is None:
            raise RuntimeError("ADK AgentRuntime is not initialized.")

        # This is the core ADK A2A call
        adk_response_message = await agent_runtime.send_message(
            agent_id=target_peer_id,
            message=adk_message_to_send
        )

        logger.info(f"Agent '{target_peer_id}' responded: {adk_response_message.text[:100]}...")
        return adk_response_message.text
    except Exception as e:
        logger.error(f"Error invoking agent '{target_peer_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to invoke agent '{target_peer_id}': {e}")

@mcp.tool()
async def list_agents(capabilities: Optional[List[str]] = None) -> str:
    """
    Lists all registered agents, optionally filtered by capabilities.
    Returns a JSON string of agent cards.
    """
    logger.info(f"Listing agents with capabilities: {capabilities if capabilities else 'all'}")

    filtered_agents = []
    for peer_id, agent_info in _registered_agents.items():
        if capabilities:
            # Check if all requested capabilities are present
            if all(cap in agent_info["capabilities"] for cap in capabilities):
                filtered_agents.append({
                    "peer_id": peer_id,
                    "name": agent_info["name"],
                    "bio": agent_info["bio"],
                    "capabilities": agent_info["capabilities"]
                })
        else:
            filtered_agents.append({
                "peer_id": peer_id,
                "name": agent_info["name"],
                "bio": agent_info["bio"],
                "capabilities": agent_info["capabilities"]
            })

    return json.dumps(filtered_agents, indent=2)


http_mcp = mcp.http_app(transport="streamable-http")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for the agent-marketplace-mcp.
    Initializes the ADK AgentRuntime.
    """
    global agent_runtime
    logger.info("Starting agent-marketplace-mcp lifespan...")

    # Initialize ADK AgentRuntime
    # The store is for ADK's internal state, not our agent registry.
    agent_runtime = AgentRuntime(store=MemoryStore()) 
    logger.info("ADK AgentRuntime initialized.")

    # Optionally, if this MCP itself is an ADK agent, you would define it here.
    # For now, it's a server hosting tools for other agents.

    # Yield control to the FastAPI application
    async with http_mcp.router.lifespan_context(app) as fastmcp_lifespan_yield:
        yield fastmcp_lifespan_yield

    logger.info("Shutting down agent-marketplace-mcp. Cleaning up ADK AgentRuntime...")
    # ADK AgentRuntime doesn't have an explicit 'close' method in basic usage,
    # but you'd ensure proper cleanup of any persistent connections or resources.
    agent_runtime = None # Dereference for garbage collection
    logger.info("Agent-marketplace-mcp shutdown complete.")

app = FastAPI(lifespan=lifespan)
app.mount("/", http_mcp)
logger.info("Agent Marketplace MCP server initialized and tools registered.")
