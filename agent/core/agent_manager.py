import logging
import os
import asyncio
import base64
from typing import Any, Dict, Tuple, Optional, List
from pydantic import Field, PrivateAttr

# LangChain imports
from langchain_groq import ChatGroq
from langchain.tools import BaseTool

from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions.in_memory_session_service import InMemorySessionService 

# FastMCP imports
from langchain_mcp_adapters.client import MultiServerMCPClient

# libp2p imports
import libp2p.crypto.keys as libp2p_key # Corrected import for key generation
from libp2p.peer.id import ID as PeerID
from multiaddr import Multiaddr # Added for completeness

# Google ADK imports (updated for 1.7.0)
from google.adk.runners import Runner as AgentRuntime
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService # For AgentRuntime store
from google.adk.agents.base_agent import BaseAgent # New base for custom agents
from google.adk.agents import LlmAgent # LlmAgent is still here

# Local imports - IMPORTANT: Using bot.models.agent_config for AgentConfig and Message
from agent.db.sqlite_manager import SQLiteManager
from agent.models.agent_config import AgentConfig, Message # Assuming Message is now defined here
from agent.prompts import AGENT_SYSTEM_PROMPT
from agent.langgraph_agents.custom_tool_agent import LangGraphADKAgent # Your custom agent class


logger = logging.getLogger(__name__)

# --- Global Components ---
# This will be managed by the AgentManager class, but kept global for simplicity
# if other parts of the app need direct access.
_initialized_agents: Dict[str, Dict[str, Any]] = {}


class TelegramToolWrapper(BaseTool):
    """
    A wrapper for Telegram tools that injects API credentials into the tool's arguments.
    This allows a single Telegram MCP server to manage multiple Telegram bots.
    """
    _wrapped_tool: BaseTool = PrivateAttr()
    telegram_api_id: int = Field(..., description="Telegram API ID for the bot.")
    telegram_api_hash: str = Field(..., description="Telegram API Hash for the bot.")
    telegram_bot_token: str = Field(..., description="Telegram Bot Token.")

    def __init__(self, wrapped_tool: BaseTool, telegram_api_id: int, telegram_api_hash: str, telegram_bot_token: str, **kwargs: Any):
        super().__init__(
            name=wrapped_tool.name,
            description=wrapped_tool.description,
            args_schema=wrapped_tool.args_schema,
            return_direct=wrapped_tool.return_direct,
            func=wrapped_tool.func,
            coroutine=wrapped_tool.coroutine,
            # Pass these to Pydantic for validation
            telegram_api_id=telegram_api_id,
            telegram_api_hash=telegram_api_hash,
            telegram_bot_token=telegram_bot_token,
            **kwargs
        )
        self._wrapped_tool = wrapped_tool

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        all_kwargs = {**kwargs} 
        all_kwargs['telegram_api_id'] = self.telegram_api_id
        all_kwargs['telegram_api_hash'] = self.telegram_api_hash
        all_kwargs['telegram_bot_token'] = self.telegram_bot_token
        
        logger.debug(f"Invoking wrapped Telegram tool '{self.name}' with injected credentials. Final Args: {all_kwargs}")
        return await self._wrapped_tool.ainvoke(all_kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Telegram tools are asynchronous and should use _arun.")


class DiscordToolWrapper(BaseTool):
    """
    A wrapper for Discord tools that injects the bot_id into the tool's arguments.
    This allows a single Discord MCP server to manage multiple Discord bots.
    """
    _wrapped_tool: BaseTool = PrivateAttr()
    bot_id: str = Field(..., description="The Discord bot ID to use for this tool.")

    def __init__(self, wrapped_tool: BaseTool, bot_id: str, **kwargs: Any):
        super().__init__(
            name=wrapped_tool.name,
            description=wrapped_tool.description,
            args_schema=wrapped_tool.args_schema,
            return_direct=wrapped_tool.return_direct,
            func=wrapped_tool.func,
            coroutine=wrapped_tool.coroutine,
            bot_id=bot_id, # Pass to Pydantic
            **kwargs
        )
        self._wrapped_tool = wrapped_tool

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Asynchronously runs the wrapped tool, injecting the Discord bot_id."""
        all_kwargs = {**kwargs}
        all_kwargs['bot_id'] = self.bot_id # Inject the bot_id
        
        logger.debug(f"Invoking wrapped Discord tool '{self.name}' with injected bot_id: {self.bot_id}. Final Args: {all_kwargs}")
        return await self._wrapped_tool.ainvoke(all_kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Discord tools are asynchronous and should use _arun.")


async def create_dynamic_agent_instance(agent_config: AgentConfig, local_mode: bool) -> Tuple[LangGraphADKAgent, MultiServerMCPClient, Optional[str], Optional[str]]:
    """
    Dynamically creates and initializes an agent instance based on AgentConfig.
    Returns the compiled agent executor, its associated MCPClient, and fetched bot IDs.
    The MCPClient will include tools based on the provided secrets, ensuring
    the basic toolkit (web, finance, rag) is always present.
    """
    agent_id = agent_config.id
    agent_name = agent_config.name
    agent_bio = agent_config.bio 
    agent_persona = agent_config.persona

    logger.info(f"Dynamically initializing agent '{agent_name}' (ID: {agent_id})...")

    # Initialize LLM
    groq_api_key = agent_config.secrets.groq_api_key
    if not groq_api_key:
        raise ValueError("Groq API Key is not provided in agent config.")
    llm = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        groq_api_key=groq_api_key
    )
    logger.info(f"✅ Initialized Groq LLM for agent '{agent_name}' with llama3-70b-8192")


    # --- libp2p Peer ID Generation ---
    if agent_config.id is None:
        # Generate a new libp2p keypair and PeerID
        private_key_obj,_ = await libp2p_key.create_new_key_pair("ed25519") # Corrected function call
        agent_libp2p_peer_id = str(PeerID.from_pubkey(private_key_obj.public_key))
        agent_config.id = agent_libp2p_peer_id

        # Store the private key (Base64 encoded) in secrets for potential future use
        # WARNING: In a real production system, this private key should be managed
        # securely (e.g., KMS, HashiCorp Vault) and NOT directly in AgentSecrets.
        agent_config.secrets.libp2p_private_key = base64.b64encode(private_key_obj.to_bytes()).decode('utf-8')
        logger.info(f"Generated new libp2p Peer ID for agent '{agent_name}': {agent_config.id}")
        logger.debug(f"Private Key (Base64 encoded, for dev only): {agent_config.secrets.libp2p_private_key[:10]}...") # Log snippet
    else:
        logger.info(f"Using existing libp2p Peer ID for agent '{agent_name}': {agent_config.id}")
        # If an ID is provided, assume it's a libp2p Peer ID.
        # If a private key is also provided in secrets, it can be used to re-derive the PeerID and verify.


    mcp_base_url_prefix = "http://localhost:900" if local_mode else "http://" # Assuming your Docker service names for non-local
    mcp_suffix = "/mcp/" # FastMCP endpoints typically end with /mcp/

    # Ensure these URLs match your docker-compose or local setup
    agent_mcp_config = {
        "web_mcp": {"url": os.getenv("WEB_MCP_URL", f"{mcp_base_url_prefix}0{mcp_suffix}"), "transport": "streamable_http"},
        "finance_mcp": {"url": os.getenv("FINANCE_MCP_URL", f"{mcp_base_url_prefix}1{mcp_suffix}"), "transport": "streamable_http"},
        "rag_mcp": {"url": os.getenv("RAG_MCP_URL", f"{mcp_base_url_prefix}2{mcp_suffix}"), "transport": "streamable_http"},
        "agent_marketplace": {"url": os.getenv("ADK_A2A_SERVER_URL", f"{mcp_base_url_prefix}5{mcp_suffix}"), "transport": "streamable_http"}, # Port 9006 for A2A server
    }

    discord_bot_id = None
    telegram_bot_id = None

    # Check for Discord secrets and add Discord MCP if present
    discord_secrets_provided = bool(agent_config.secrets.discord_bot_token)
    if discord_secrets_provided:
        agent_mcp_config["discord_mcp"] = {"url": os.getenv("DISCORD_MCP_URL", f"{mcp_base_url_prefix}4{mcp_suffix}"), "transport": "streamable_http"} # Port 9005 for Discord
        logger.info(f"Agent '{agent_name}' will include Discord tools.")
    else:
        logger.info(f"Agent '{agent_name}' does not have Discord bot token. Discord tools will NOT be enabled.")

    # Check for Telegram secrets and add Telegram MCP if present
    telegram_secrets_provided = (
        agent_config.secrets.telegram_bot_token and
        agent_config.secrets.telegram_api_id is not None and # Check for None explicitly
        agent_config.secrets.telegram_api_hash
    )
    if telegram_secrets_provided:
        agent_mcp_config["telegram_mcp"] = {"url": os.getenv("TELEGRAM_MCP_URL", f"{mcp_base_url_prefix}3{mcp_suffix}"), "transport": "streamable_http"} # Port 9004 for Telegram
        logger.info(f"Agent '{agent_name}' will include Telegram tools.")
    else:
        if agent_config.secrets.telegram_bot_token:
            logger.warning(f"Agent '{agent_name}' has Telegram bot token but is missing telegram_api_id or telegram_api_hash. Telegram tools will NOT be enabled.")


    mcp_client = MultiServerMCPClient(agent_mcp_config)
    mcp_client.tools = {} # Initialize mcp_client.tools before populating

    agent_tools_raw = []
    agent_tools_final = []

    logger.info(f"Attempting to load tools for agent '{agent_name}' from MCP servers: {list(agent_mcp_config.keys())}...")
    try:
        fetched_tools_list = await mcp_client.get_tools()
        if fetched_tools_list:
            agent_tools_raw = list(fetched_tools_list)
            
            # First, handle Discord bot registration if token is provided
            if discord_secrets_provided:
                register_discord_tool = next((t for t in agent_tools_raw if t.name == "register_discord_bot"), None)
                if register_discord_tool:
                    try:
                        logger.info(f"Calling 'register_discord_bot' for agent '{agent_name}' with token (first 5 chars): {agent_config.secrets.discord_bot_token[:5]}...")
                        # The register_discord_bot tool returns the bot_id
                        discord_bot_id = await register_discord_tool.ainvoke({"bot_token": agent_config.secrets.discord_bot_token})
                        logger.info(f"Successfully registered Discord bot for agent '{agent_name}'. Bot ID: {discord_bot_id}")
                    except Exception as e:
                        logger.error(f"Failed to register Discord bot for agent '{agent_name}': {e}", exc_info=True)
                        discord_bot_id = None # Ensure it's None if registration fails
                else:
                    logger.warning(f"Agent '{agent_name}' has Discord token but 'register_discord_bot' tool not found. Discord tools will NOT be enabled.")
            
            # Now, process and wrap all tools
            for tool_item in agent_tools_raw: # Renamed 'tool' to 'tool_item' to avoid shadowing built-in
                if telegram_secrets_provided and tool_item.name in ["send_message_telegram", "get_chat_history", "get_bot_id_telegram"]:
                    logger.debug(f"Wrapping Telegram tool '{tool_item.name}' for agent '{agent_name}'.")
                    try:
                        telegram_api_id_int = int(agent_config.secrets.telegram_api_id)
                    except (ValueError, TypeError): # Handle both ValueError for int() and TypeError for None
                        logger.error(f"Invalid or missing telegram_api_id for agent '{agent_name}': {agent_config.secrets.telegram_api_id}. Skipping Telegram tool wrapping.")
                        agent_tools_final.append(tool_item)
                        mcp_client.tools[tool_item.name] = tool_item
                        continue

                    wrapped_tool = TelegramToolWrapper(
                        wrapped_tool=tool_item,
                        telegram_api_id=telegram_api_id_int,
                        telegram_api_hash=agent_config.secrets.telegram_api_hash,
                        telegram_bot_token=agent_config.secrets.telegram_bot_token
                    )
                    agent_tools_final.append(wrapped_tool)
                    mcp_client.tools[wrapped_tool.name] = wrapped_tool 
                
                elif discord_bot_id and tool_item.name in ["send_message", "get_channel_messages", "get_bot_id"]:
                    logger.debug(f"Wrapping Discord tool '{tool_item.name}' for agent '{agent_name}' with bot ID: {discord_bot_id}.")
                    wrapped_tool = DiscordToolWrapper(
                        wrapped_tool=tool_item,
                        bot_id=discord_bot_id
                    )
                    agent_tools_final.append(wrapped_tool)
                    mcp_client.tools[wrapped_tool.name] = wrapped_tool
                
                else:
                    agent_tools_final.append(tool_item)
                    mcp_client.tools[tool_item.name] = tool_item
            
            if telegram_secrets_provided and "telegram_mcp" in agent_mcp_config: # Use telegram_mcp key
                get_telegram_bot_id_tool = mcp_client.tools.get("get_bot_id_telegram")
                if get_telegram_bot_id_tool:
                    try:
                        telegram_bot_id = await get_telegram_bot_id_tool.ainvoke({})
                        logger.info(f"Fetched Telegram Bot ID for agent '{agent_name}': {telegram_bot_id}")
                    except Exception as e:
                        logger.warning(f"Failed to fetch Telegram Bot ID for agent '{agent_name}': {e}", exc_info=True)
            
            # --- Agent Self-Registration with Agent Marketplace MCP (ADK A2A Server) ---
            register_tool = mcp_client.tools.get("register_agent_capability")
            if register_tool:
                try:
                    agent_capabilities = ["chat", "general_query"]
                    # Add capabilities based on MCPs included in agent_mcp_config
                    if "web_mcp" in agent_mcp_config:
                        agent_capabilities.append("web_search")
                    if "finance_mcp" in agent_mcp_config:
                        agent_capabilities.append("finance_queries")
                    if "rag_mcp" in agent_mcp_config:
                        agent_capabilities.append("knowledge_retrieval")
                    if "telegram_mcp" in agent_mcp_config and telegram_bot_id: # Only add if Telegram bot is actually registered
                        agent_capabilities.append("telegram_bot")
                    if "discord_mcp" in agent_mcp_config and discord_bot_id: # Only add if Discord bot is actually registered
                        agent_capabilities.append("discord_bot")

                    # Internal URL for this specific cyrene-agent instance to be invoked by A2A server
                    # This needs to be accessible by the agent-marketplace-mcp.
                    # Use environment variable for flexibility.
                    internal_invoke_url = f"{os.getenv('CYRENE_AGENT_INTERNAL_URL', 'http://localhost:8000')}/internal/agents/{agent_config.id}"

                    logger.info(f"Agent '{agent_name}' (ID: {agent_config.id}) registering with agent marketplace at {internal_invoke_url}...")
                    registration_result = await register_tool.ainvoke({
                        "peer_id": agent_config.id,
                        "name": agent_name,
                        "bio": agent_config.bio,
                        "capabilities": agent_capabilities,
                        "internal_url": internal_invoke_url,
                        "framework": "langgraph" # Specify framework for cross-framework A2A
                    })
                    logger.info(f"Agent registration result: {registration_result}")
                except Exception as e:
                    logger.error(f"Failed to register agent '{agent_name}' with agent marketplace: {e}", exc_info=True)
            else:
                logger.warning(f"Agent '{agent_name}' could not find 'register_agent_capability' tool. Agent will not be registered in marketplace.")

        else:
            logger.warning(f"No tools fetched for agent '{agent_name}'. This might mean configured MCP servers are down or no tools are exposed for configured services.")
    except Exception as e:
        logger.error(f"Error loading tools for agent '{agent_name}': {e}", exc_info=True)
        if not hasattr(mcp_client, 'tools'):
            mcp_client.tools = {}
        agent_tools_final = []

    logger.info(f"🔧 Loaded {len(agent_tools_final)} tools for agent '{agent_name}'. Tools found: {[t.name for t in agent_tools_final]}.")
    logger.info(f"Final number of tools obtained for agent '{agent_name}': {len(agent_tools_final)}")

    system_prompt = AGENT_SYSTEM_PROMPT
    if agent_config.persona:
        system_prompt = f"{system_prompt}\n\nYour persona: {agent_config.persona}"
    if agent_config.bio:
        system_prompt = f"{system_prompt}\n\nYour bio: {agent_config.bio}"
    if agent_config.knowledge:
        system_prompt = f"{system_prompt}\n\nKnowledge: {system_prompt}" # Typo fix: should be agent_config.knowledge
    logger.info(f"Using AGENT_SYSTEM_PROMPT (or extended) for agent '{agent_name}'.")


    logger.info(f"ADK AgentRuntime initialized for agent '{agent_name}'.")

    current_agent_instance = LangGraphADKAgent(
        agent_id=agent_config.id, # Pass the PeerID
        agent_name=agent_name,
        agent_bio=agent_bio,
        agent_persona=agent_config.persona, # Pass persona
        llm=llm,
        mcp_client=mcp_client,
        tools=agent_tools_final,
        adk_runtime=None # Pass the ADK runtime
    )

    agent_adk_runtime = AgentRuntime(
            app_name=f"cyrene-agent-{agent_name}", # Unique app name
            agent=current_agent_instance, # Pass the LangGraphADKAgent instance
            session_service=InMemorySessionService(), # Provide a session service
            memory_service=InMemoryMemoryService() # Provide a memory service
        )
    
    current_agent_instance.adk_runtime = agent_adk_runtime

    # Store the initialized agent details in the global cache
    _initialized_agents[agent_config.id] = {
        "name": agent_name,
        "executor": current_agent_instance,
        "mcp_client": mcp_client,
        "discord_bot_id": discord_bot_id,
        "telegram_bot_id": telegram_bot_id,
        "agent_config": agent_config # Store the full config object
    }
    logger.info(f"🧠 Agent: {agent_name} (ID: {agent_config.id}) initialized as a custom LangGraph agent in ADK with {len(agent_tools_final)} tools.")
    return current_agent_instance, mcp_client, discord_bot_id, telegram_bot_id

class AgentManager:
    def __init__(self, db_path: str):
        # Assuming db_manager is initialized elsewhere or passed in
        # For now, we'll use a placeholder or assume it's set globally if needed by other functions
        # from bot.db.sqlite_manager import SQLiteManager # Uncomment if needed
        # self.db_manager = SQLiteManager(db_path) # Uncomment if needed
        # self.db_manager.create_tables() # Uncomment if needed
        self.db_path = db_path # Store db_path for reference
        logger.info(f"AgentManager initialized (DB path: {db_path}).")

    async def initialize_all_agents_from_db(self, local_mode: bool = True):
        """Initializes all agents stored in the database."""
        db_manager_instance = SQLiteManager(self.db_path)
        db_manager_instance.create_tables() # Ensure tables exist

        agents = db_manager_instance.get_all_agent_configs()
        logger.info(f"Found {len(agents)} agents in DB. Initializing...")
        for agent_config_dict in agents:
            agent_config = AgentConfig(**agent_config_dict)
            if agent_config.id not in _initialized_agents:
                logger.info(f"Initializing agent '{agent_config.name}' (ID: {agent_config.id})...")
                try:
                    # Call the global/module-level create_dynamic_agent_instance
                    executor, mcp_client, discord_id, telegram_id = await create_dynamic_agent_instance(agent_config, local_mode)
                    # Add to the global cache
                    _initialized_agents[agent_config.id] = {
                        "name": agent_config.name,
                        "executor": executor,
                        "mcp_client": mcp_client,
                        "discord_bot_id": discord_id,
                        "telegram_bot_id": telegram_id,
                        "agent_config": agent_config
                    }
                    logger.info(f"Agent '{agent_config.name}' (ID: {agent_config.id}) initialized and cached.")
                except Exception as e:
                    logger.error(f"Failed to initialize agent '{agent_config.name}' (ID: {agent_config.id}): {e}", exc_info=True)
            else:
                logger.info(f"Agent '{agent_config.name}' (ID: {agent_config.id}) already initialized.")

    def get_initialized_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an initialized agent (executor and mcp_client) from the cache."""
        return _initialized_agents.get(agent_id)

    def get_all_initialized_agents(self) -> Dict[str, Dict[str, Any]]:
        """Returns all initialized agents."""
        return _initialized_agents

    def get_agent_executor(self, agent_id: str) -> Any:
        """Returns the LangGraph agent executor for a given agent ID."""
        agent_info = self.get_initialized_agent(agent_id)
        return agent_info.get("executor") if agent_info else None

    def get_agent_mcp_client(self, agent_id: str) -> Optional[MultiServerMCPClient]:
        """Returns the MCP client for a given agent ID."""
        agent_info = self.get_initialized_agent(agent_id)
        return agent_info.get("mcp_client") if agent_info else None

    def get_agent_config(self, agent_id: str) -> Optional[AgentConfig]:
        """Returns the AgentConfig for a given agent ID."""
        agent_info = self.get_initialized_agent(agent_id)
        return agent_info.get("agent_config") if agent_info else None

    async def shutdown_all_agents(self):
        """Shuts down all initialized agents and their components."""
        logger.info("Shutting down all agents...")
        # Iterate over a copy of items because we're modifying the dict
        for agent_id, agent_info in list(_initialized_agents.items()):
            mcp_client = agent_info.get("mcp_client")
            if mcp_client:
                await mcp_client.close()
                logger.info(f"MCP Client for agent {agent_id} closed.")
            
            # If the agent_executor has an ADK runtime, close it
            agent_executor = agent_info.get("executor")
            if hasattr(agent_executor, 'adk_runtime') and agent_executor.adk_runtime:
                # ADK AgentRuntime has a `close` method for cleanup
                await agent_executor.adk_runtime.close()
                logger.info(f"ADK Runtime for agent {agent_id} closed.")

            _initialized_agents.pop(agent_id)
            logger.info(f"Agent {agent_id} removed from registry.")
        logger.info("All agents shut down.")

