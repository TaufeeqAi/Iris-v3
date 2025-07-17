import logging
from typing import Any, Dict, Tuple, Optional
from pydantic import Field, PrivateAttr

from langchain_groq import ChatGroq
from langchain.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.models.agent_config import AgentConfig
from agent.prompts import AGENT_SYSTEM_PROMPT
from agent.langgraph_agents.custom_tool_agent import create_custom_tool_agent,LangGraphADKAgent


import base64
from libp2p.crypto.keys import generate_key_pair
from libp2p.peer.id import ID as PeerID
from multiaddr import Multiaddr 
from google.adk.agents import Message # For ADK message payload


logger = logging.getLogger(__name__)

_initialized_agents: Dict[str, Dict[str, Any]] = {}

def add_initialized_agent(
        agent_id: str, 
        agent_name: str, 
        executor: Any, 
        mcp_client: MultiServerMCPClient, 
        discord_bot_id: Optional[str] = None, 
        telegram_bot_id: Optional[str] = None):
    """Adds an initialized agent, its MCP client, and platform-specific bot IDs to the cache."""
    agent_info = {
        "name": agent_name,
        "executor": executor,
        "mcp_client": mcp_client
    }
    if discord_bot_id:
        agent_info["discord_bot_id"] = discord_bot_id
    if telegram_bot_id:
        agent_info["telegram_bot_id"] = telegram_bot_id
        
    _initialized_agents[agent_id] = agent_info
    logger.info(f"Agent '{agent_name}' (ID: {agent_id}) and its MCP client added to cache. Discord Bot ID: {discord_bot_id}, Telegram Bot ID: {telegram_bot_id}")

def get_initialized_agent(agent_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves an initialized agent (executor and mcp_client) from the cache."""
    return _initialized_agents.get(agent_id)

def clear_initialized_agents_cache():
    """Clears all initialized agents from the cache."""
    _initialized_agents.clear()
    logger.info("Initialized agents cache cleared.")


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
    logger.info(f"Dynamically initializing agent '{agent_name}' (ID: {agent_id})...")

    llm = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        groq_api_key=agent_config.secrets.groq_api_key
    )
    logger.info(f"✅ Initialized Groq LLM for agent '{agent_name}' with llama3-8b-8192")


  # --- libp2p Peer ID Generation ---
    if agent_config.id is None:
        # Generate a new libp2p keypair and PeerID
        private_key, public_key = generate_key_pair('ed25519')
        agent_libp2p_peer_id = str(PeerID.from_pubkey(public_key))
        agent_config.id = agent_libp2p_peer_id

        # Store the private key (Base64 encoded) in secrets for potential future use
        # WARNING: In a real production system, this private key should be managed
        # securely (e.g., KMS, HashiCorp Vault) and NOT directly in AgentSecrets.
        agent_config.secrets.libp2p_private_key = base64.b64encode(private_key.to_bytes()).decode('utf-8')
        logger.info(f"Generated new libp2p Peer ID for agent '{agent_name}': {agent_config.id}")
        logger.debug(f"Private Key (Base64 encoded, for dev only): {agent_config.secrets.libp2p_private_key[:10]}...") # Log snippet
    else:
        logger.info(f"Using existing libp2p Peer ID for agent '{agent_name}': {agent_config.id}")
        # If an ID is provided, assume it's a libp2p Peer ID.
        # If a private key is also provided in secrets, it can be used to re-derive the PeerID and verify.



    mcp_base_url_prefix = "http://localhost:900" if local_mode else "http://"
    mcp_suffix = "/mcp/" if local_mode else "/mcp"

    agent_mcp_config = {
        "multi_search": {"url": f"{mcp_base_url_prefix}0{mcp_suffix}", "transport": "streamable_http"},
        "finance": {"url": f"{mcp_base_url_prefix}1{mcp_suffix}", "transport": "streamable_http"},
        "rag": {"url": f"{mcp_base_url_prefix}2{mcp_suffix}", "transport": "streamable_http"},
        "agent_marketplace": {"url": f"{mcp_base_url_prefix}6{mcp_suffix}", "transport": "streamable_http"},
    }

    discord_bot_id = None
    telegram_bot_id = None

    # Check for Discord secrets and add Discord MCP if present
    discord_secrets_provided = bool(agent_config.secrets.discord_bot_token)
    if discord_secrets_provided:
        agent_mcp_config["discord"] = {"url": f"{mcp_base_url_prefix}4{mcp_suffix}", "transport": "streamable_http"}
        logger.info(f"Agent '{agent_name}' will include Discord tools.")
    else:
        logger.info(f"Agent '{agent_name}' does not have Discord bot token. Discord tools will NOT be enabled.")

    # Check for Telegram secrets and add Telegram MCP if present
    telegram_secrets_provided = (
        agent_config.secrets.telegram_bot_token and
        agent_config.secrets.telegram_api_id and
        agent_config.secrets.telegram_api_hash
    )
    if telegram_secrets_provided:
        agent_mcp_config["telegram"] = {"url": f"{mcp_base_url_prefix}3{mcp_suffix}", "transport": "streamable_http"}
        logger.info(f"Agent '{agent_name}' will include Telegram tools.")
    else:
        if agent_config.secrets.telegram_bot_token:
            logger.warning(f"Agent '{agent_name}' has Telegram bot token but is missing telegram_api_id or telegram_api_hash. Telegram tools will NOT be enabled.")


    mcp_client = MultiServerMCPClient(agent_mcp_config)
    mcp_client.tools = {} # Initialize mcp_client.tools before populating

    agent_tools_raw = []
    agent_tools_final = []

    logger.info(f"Attempt 1/15: Loading tools for agent '{agent_name}' from MCP servers at {list(agent_mcp_config.keys())}...")
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
            for tool in agent_tools_raw:
                if telegram_secrets_provided and tool.name in ["send_message_telegram", "get_chat_history", "get_bot_id_telegram"]:
                    logger.debug(f"Wrapping Telegram tool '{tool.name}' for agent '{agent_name}'.")
                    try:
                        telegram_api_id_int = int(agent_config.secrets.telegram_api_id)
                    except ValueError:
                        logger.error(f"Invalid telegram_api_id for agent '{agent_name}': {agent_config.secrets.telegram_api_id}. Skipping Telegram tool wrapping.")
                        agent_tools_final.append(tool)
                        mcp_client.tools[tool.name] = tool
                        continue

                    wrapped_tool = TelegramToolWrapper(
                        wrapped_tool=tool,
                        telegram_api_id=telegram_api_id_int,
                        telegram_api_hash=agent_config.secrets.telegram_api_hash,
                        telegram_bot_token=agent_config.secrets.telegram_bot_token
                    )
                    agent_tools_final.append(wrapped_tool)
                    mcp_client.tools[wrapped_tool.name] = wrapped_tool 
                
                elif discord_bot_id and tool.name in ["send_message", "get_channel_messages", "get_bot_id"]:
                    logger.debug(f"Wrapping Discord tool '{tool.name}' for agent '{agent_name}' with bot ID: {discord_bot_id}.")
                    wrapped_tool = DiscordToolWrapper(
                        wrapped_tool=tool,
                        bot_id=discord_bot_id
                    )
                    agent_tools_final.append(wrapped_tool)
                    mcp_client.tools[wrapped_tool.name] = wrapped_tool
                
                else:
                    agent_tools_final.append(tool)
                    mcp_client.tools[tool.name] = tool
            
            if telegram_secrets_provided and "telegram" in agent_mcp_config:
                get_telegram_bot_id_tool = mcp_client.tools.get("get_bot_id_telegram")
                if get_telegram_bot_id_tool:
                    try:
                        telegram_bot_id = await get_telegram_bot_id_tool.ainvoke({})
                        logger.info(f"Fetched Telegram Bot ID for agent '{agent_name}': {telegram_bot_id}")
                    except Exception as e:
                        logger.warning(f"Failed to fetch Telegram Bot ID for agent '{agent_name}': {e}", exc_info=True)
            
            # --- Agent Self-Registration with Agent Marketplace MCP ---
            register_tool = next((t for t in agent_tools_raw if t.name == "register_agent_capability"), None)
            if register_tool:
                try:
                    agent_capabilities = ["chat", "general_query"]
                    if "finance" in agent_mcp_config:
                        agent_capabilities.append("finance_queries")
                    if "rag" in agent_mcp_config:
                        agent_capabilities.append("knowledge_retrieval")
                    if "telegram" in agent_mcp_config:
                        agent_capabilities.append("telegram_bot")
                    if "discord" in agent_mcp_config:
                        agent_capabilities.append("discord_bot")

                    internal_invoke_url = f"http://cyrene-agent:8000/internal/agents/{agent_config.id}"

                    logger.info(f"Agent '{agent_name}' (ID: {agent_config.id}) registering with agent marketplace at {internal_invoke_url}...")
                    registration_result = await register_tool.ainvoke({
                        "peer_id": agent_config.id,
                        "name": agent_name,
                        "bio": agent_config.bio,
                        "capabilities": agent_capabilities,
                        "internal_url": internal_invoke_url
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
        system_prompt = f"{system_prompt}\n\nKnowledge: {agent_config.knowledge}" 
    logger.info(f"Using AGENT_SYSTEM_PROMPT (or extended) for agent '{agent_name}'.")

    agent_executor = await create_custom_tool_agent(llm, agent_tools_final, system_prompt, agent_name, agent_id)

    logger.info(f"🧠 Agent: {agent_name} (ID: {agent_config.id}) initialized as a custom LangGraph agent in ADK with {len(agent_tools_final)} tools.")
    return agent_executor, mcp_client, discord_bot_id, telegram_bot_id

