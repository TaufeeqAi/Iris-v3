import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from dotenv import load_dotenv

# For internal HTTP calls to cyrene-agent instances
import httpx
import json

# Explicitly import all libp2p components needed
from libp2p import new_host
from libp2p.crypto.keys import KeyPair
from libp2p.peer.id import ID as PeerID 
from multiaddr import Multiaddr
from libp2p.security.noise.transport import Transport as NoiseTransport
from libp2p.stream_muxer.mplex.mplex import Mplex
from libp2p.transport.tcp.tcp import TCP 
from libp2p.security.security_multistream import SecurityMultistream


# Import your common models, now from bot.models.agent_config
from agent.models.agent_config import AgentConfig, AgentSecrets, Message

# Import your custom key generation function
from common.libp2p_utils import generate_libp2p_keypair

load_dotenv()

logger = logging.getLogger(__name__)
try:
    from common.utils import setup_logging
    setup_logging(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.warning("common.utils.setup_logging not found, using basic logging config.")

mcp = FastMCP("agent_marketplace")

# --- Google ADK imports for 1.7.0 (Based on your provided package overview) ---
from google.adk.runners import Runner as AgentRuntime
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService


# --- Global ADK AgentRuntime ---
agent_runtime: Optional[AgentRuntime] = None

# --- Agent Registry (in-memory for now, can be persistent later) ---
_registered_agents: Dict[str, Dict[str, Any]] = {}

_libp2p_host: Optional[Any] = None 
_kademlia_dht = None 
_libp2p_keypair: Optional[KeyPair] = None 

# --- Custom HTTP Connectors ---
class CyreneAgentConnector:
    """
    Custom connector for communicating with a cyrene-agent (LangGraph-based) via HTTP.
    It sends/receives `bot.models.agent_config.Message` objects.
    """
    def __init__(self, agent_id: str, internal_url: str):
        self.agent_id = agent_id
        self.internal_url = internal_url
        logger.info(f"Initialized CyreneAgentConnector for agent {agent_id} at {internal_url}")

    async def send(self, message: Message) -> Message:
        """
        Sends a message to the target cyrene-agent instance's internal ADK endpoint.
        Expects and returns `bot.models.agent_config.Message`.
        """
        logger.info(f"CyreneAgentConnector sending message to {self.internal_url} for agent {self.agent_id}")
        try:
            adk_message_for_cyrene = message.model_dump() if hasattr(message, 'model_dump') else message.dict()

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.internal_url}/invoke_adk",
                    json=adk_message_for_cyrene,
                    timeout=60.0
                )
                response.raise_for_status()
                
                response_data = response.json()
                returned_message = Message(**response_data)
                return returned_message

        except httpx.HTTPStatusError as e:
            logger.error(f"CyreneAgentConnector HTTP error for {self.agent_id} at {self.internal_url}: {e.response.status_code} - {e.response.text}", exc_info=True)
            raise HTTPException(status_code=e.response.status_code, detail=f"A2A communication error: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"CyreneAgentConnector request error for {self.agent_id} at {self.internal_url}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"A2A communication error: {e}")
        except Exception as e:
            logger.error(f"CyreneAgentConnector unexpected error for {self.agent_id} at {self.internal_url}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"A2A communication error: {e}")


class NodeAgentConnector:
    """
    Custom connector for communicating with a Node.js agent via HTTP.
    It sends/receives `bot.models.agent_config.Message` objects.
    """
    def __init__(self, agent_id: str, internal_url: str):
        self.agent_id = agent_id
        self.internal_url = internal_url
        logger.info(f"Initialized NodeAgentConnector for agent {agent_id} at {internal_url}")

    async def send(self, message: Message) -> Message:
        """
        Sends a message to the target Node.js agent's internal endpoint.
        Expects and returns `bot.models.agent_config.Message`.
        """
        logger.info(f"NodeAgentConnector sending message to {self.internal_url} for agent {self.agent_id}")
        try:
            adk_message_for_node = message.model_dump() if hasattr(message, 'model_dump') else message.dict()

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.internal_url}/invoke",
                    json=adk_message_for_node,
                    timeout=60.0
                )
                response.raise_for_status()
                response_data = response.json()
                returned_message = Message(**response_data)
                return returned_message
        except Exception as e:
            logger.error(f"NodeAgentConnector error for {self.agent_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Node.js A2A communication error: {e}")


# --- FastMCP Tools Exposed by agent-marketplace-mcp ---
@mcp.tool()
async def register_agent_capability(
    peer_id: str,
    name: str,
    bio: str,
    capabilities: List[str],
    internal_url: str,
    framework: str = "langgraph"
) -> str:
    """
    Registers an agent's capabilities with the ADK A2A server.
    NOTE: Kademlia DHT publication is currently disabled as it's not implemented in py-libp2p.
    """
    logger.info(f"Registering agent '{name}' (PeerID: {peer_id}, Framework: {framework}) with capabilities: {capabilities} at internal URL: {internal_url}")
    
    agent_card_data = {
        "peer_id": peer_id,
        "name": name,
        "bio": bio,
        "capabilities": capabilities,
        "internal_url": internal_url,
        "framework": framework
    }
    
    _registered_agents[peer_id] = agent_card_data

    if _kademlia_dht:
        logger.warning("Kademlia DHT operations are disabled as it's not implemented in py-libp2p.")
    else:
        logger.warning("Kademlia DHT not initialized (or disabled), skipping DHT publication.")

    logger.info(f"Agent '{name}' (PeerID: {peer_id}) successfully registered in agent marketplace (local registry).")
    return f"Agent '{name}' (PeerID: {peer_id}) registered successfully."


@mcp.tool()
async def list_agents_decentralized(capabilities: Optional[List[str]] = None) -> str:
    """
    Lists agents from the local registry. Decentralized discovery via Kademlia DHT is currently disabled.
    Returns a JSON string of AgentCard objects.
    """
    logger.info(f"Querying local registry for agents with capabilities: {capabilities if capabilities else 'all'}")

    if _kademlia_dht:
        logger.warning("Kademlia DHT operations are disabled, returning from local registry only.")

    found_agent_cards = []
    if capabilities:
        for peer_id, agent_data in _registered_agents.items():
            agent_capabilities = agent_data.get("capabilities", [])
            if all(cap in agent_capabilities for cap in capabilities):
                found_agent_cards.append(agent_data)
    else:
        found_agent_cards = list(_registered_agents.values())

    logger.info(f"Discovered {len(found_agent_cards)} agents via local registry.")
    return json.dumps(found_agent_cards, indent=2)


@mcp.tool()
async def invoke_agent(caller_peer_id: str, target_peer_id: str, message: str) -> str:
    """
    Invokes another agent by its libp2p Peer ID with a given message.
    This tool is called by a cyrene-agent when it needs to delegate a task.
    """
    logger.info(f"Agent '{caller_peer_id}' invoking agent '{target_peer_id}' with message: {message[:100]}...")

    target_agent_info = _registered_agents.get(target_peer_id)
    if not target_agent_info:
        logger.error(f"Target agent with PeerID '{target_peer_id}' not found in registry.")
        raise HTTPException(status_code=404, detail=f"Target agent '{target_peer_id}' not found.")

    try:
        adk_message_to_send = Message(text=message, metadata={"caller_id": caller_peer_id})
        
        framework = target_agent_info.get("framework", "langgraph")
        internal_url = target_agent_info.get("internal_url")

        connector = None
        if framework == "langgraph":
            connector = CyreneAgentConnector(agent_id=target_peer_id, internal_url=internal_url)
        elif framework == "nodejs":
            connector = NodeAgentConnector(agent_id=target_peer_id, internal_url=internal_url)
        else:
            raise ValueError(f"Unsupported agent framework: {framework}")

        if connector is None:
            raise RuntimeError(f"Could not create connector for framework: {framework}")

        adk_response_message = await connector.send(adk_message_to_send)
        
        logger.info(f"Agent '{target_peer_id}' responded: {adk_response_message.text[:100]}...")
        return adk_response_message.text
    except Exception as e:
        logger.error(f"Error invoking agent '{target_peer_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to invoke agent '{target_peer_id}': {e}")


http_mcp = mcp.http_app(transport="streamable-http")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_runtime, _libp2p_host, _kademlia_dht, _libp2p_keypair
    logger.info("Starting agent-marketplace-mcp lifespan...")

    # --- libp2p PeerID Generation (without network initialization) ---
    try:
        _libp2p_keypair = generate_libp2p_keypair()
        host_peer_id = PeerID.from_pubkey(_libp2p_keypair.public_key) 
        logger.info(f"Agent Marketplace MCP libp2p PeerID: {host_peer_id}")
        
        # For now, just generate the PeerID without creating a network host
        # This avoids the Trio/asyncio compatibility issues
        logger.info("libp2p PeerID generated successfully")
        logger.warning("libp2p networking disabled due to Trio/asyncio compatibility issues")
        logger.warning("P2P discovery will use local registry only")
        
        # Set host to None to indicate networking is disabled
        _libp2p_host = None

    except Exception as e:
        logger.error(f"Failed to generate libp2p PeerID: {e}", exc_info=True)
        logger.warning("Continuing without libp2p - using fallback agent identification")
        _libp2p_host = None
        _libp2p_keypair = None

    async with http_mcp.router.lifespan_context(app) as fastmcp_lifespan_yield:
        yield fastmcp_lifespan_yield

    logger.info("Shutting down agent-marketplace-mcp. Cleaning up...")
    # No network host to close
    agent_runtime = None
    logger.info("Agent-marketplace-mcp shutdown complete.")

    
# --- FastAPI App Mounting ---
app = FastAPI(lifespan=lifespan)
app.mount("/", http_mcp)
logger.info("Agent Marketplace MCP server initialized and tools registered.")
