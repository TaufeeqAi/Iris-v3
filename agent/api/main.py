import os
import sys
import asyncio
import logging
import json 
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

# --- ADK imports ---
from google.adk.agents import Message # For ADK message payload
from agent.core.agent_manager import LangGraphADKAgent # Import the new class

from agent.models.agent_config import AgentConfig
from agent.db import sqlite_manager
from agent.core import agent_manager
from agent.prompts import AGENT_SYSTEM_PROMPT
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage


# --------- Load environment variables ---------
load_dotenv()

# --------- Logging Setup ---------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --------- Pydantic Model for Discord Message Payload ---------
class ReceiveDiscordMessageRequest(BaseModel):
    content: str
    channel_id: str
    author_id: str
    author_name: str
    message_id: str
    timestamp: str
    guild_id: Optional[str] = None
    bot_id: str # The ID of the bot that received the message

# --------- Determine Local or Cluster Mode ---------
LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"
logger.info(f"Running in LOCAL_MODE: {LOCAL_MODE}")

# --------- FastAPI Lifespan Context Manager ---------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for initializing and cleaning up resources.
    Initializes the SQLite database and agents.
    No global MCP client for webhooks; agents handle their own platform replies.
    """
    logger.info("Agent app startup: Initializing global resources...")
    
    # Initialize SQLite Manager instance
    db_path = os.getenv("SQLITE_DB_PATH", "agents.db")
    app.state.db_manager = sqlite_manager.SQLiteManager(db_path) # Instantiate the class
    # The database initialization (init_db) is now handled automatically within the SQLiteManager's __init__
    logger.info(f"SQLite database '{db_path}' initialized by SQLiteManager.")

    # --- Initialize or re-initialize agents ---
    # Agents will create their own MCPClient instances internally based on their secrets.
    existing_agents = await app.state.db_manager.get_all_agent_configs() 
    
    # If no agents exist, create a default agent (Streamlit-only)
    if not existing_agents:
        default_agent_config = AgentConfig(
            name="DefaultBot",
            bio="A general-purpose AI assistant for basic queries, available only on Streamlit.",
            persona="helpful and concise",
            secrets={"groq_api_key": os.getenv("GROQ_API_KEY")} 
        )
        try:
            default_agent_id = await app.state.db_manager.save_agent_config(default_agent_config)
            default_agent_config.id = default_agent_id
            
            default_executor, default_agent_mcp_client, discord_bot_id, telegram_bot_id = \
                await agent_manager.create_dynamic_agent_instance(default_agent_config, LOCAL_MODE)
            
            # Store all relevant info in the manager's cache
            agent_manager.add_initialized_agent(
                default_agent_id, 
                default_agent_config.name, 
                default_executor, 
                default_agent_mcp_client,
                discord_bot_id=discord_bot_id,
                telegram_bot_id=telegram_bot_id
            )
            logger.info("Default agent 'DefaultBot' initialized and saved to DB (Streamlit-only).")
        except Exception as e:
            logger.error(f"Failed to initialize and save default agent: {e}", exc_info=True)
    else:
        # Re-initialize existing agents from DB
        for config in existing_agents:
            try:
                executor, agent_mcp_client, discord_bot_id, telegram_bot_id = \
                    await agent_manager.create_dynamic_agent_instance(config, LOCAL_MODE)
                
                # Store all relevant info in the manager's cache
                agent_manager.add_initialized_agent(
                    config.id, 
                    config.name, 
                    executor, 
                    agent_mcp_client,
                    discord_bot_id=discord_bot_id, 
                    telegram_bot_id=telegram_bot_id
                )
            except Exception as e:
                logger.error(f"Failed to re-initialize agent '{config.name}' (ID: {config.id}): {e}", exc_info=True)

    logger.info("Agent app startup complete. Agent is ready.")
    yield
    
    # --- Shutdown resources ---
    logger.info("Agent app shutdown.")
    agent_manager.clear_initialized_agents_cache()
    # The close method on SQLiteManager is now a no-op but calling it is harmless
    app.state.db_manager.close() 
    logger.info("SQLite database connection closed.")

# --------- Main FastAPI Application Instance ---------
app = FastAPI(lifespan=lifespan)

# --------- FastAPI Endpoints ---------

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Multi-Agent Bot API!"}


# --------- Agent CRUD ---------

@app.post("/agents/create", response_model=AgentConfig, status_code=status.HTTP_201_CREATED)
async def create_agent(agent_config: AgentConfig):
    try:
        agent_id = await app.state.db_manager.save_agent_config(agent_config) # Await the async method
        agent_config.id = agent_id

        executor, mcp_client, discord_bot_id, telegram_bot_id = \
            await agent_manager.create_dynamic_agent_instance(agent_config, LOCAL_MODE)
        
        # Store all relevant info in the manager's cache
        agent_manager.add_initialized_agent(
            agent_id, 
            agent_config.name, 
            executor, 
            mcp_client,
            discord_bot_id=discord_bot_id, 
            telegram_bot_id=telegram_bot_id
        )

        return agent_config
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid agent configuration: {e.errors()}")
    except Exception as e:
        logger.error(f"Error creating agent: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create agent: {e}")

@app.get("/agents/list", response_model=List[AgentConfig])
async def list_agents():
    try:
        configs = await app.state.db_manager.get_all_agent_configs()
        return configs
    except Exception as e:
        logger.error(f"Error listing agents: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to list agents: {e}")

# --------- Streamlit Chat (Frontend) ---------

@app.post("/agents/{agent_id}/chat")
async def chat_with_agent(agent_id: str, message: Dict[str, str]):
    user_message = message.get("message")
    if not user_message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message content is required.")

    agent_info = agent_manager.get_initialized_agent(agent_id)
    if not agent_info:
        agent_config = await app.state.db_manager.get_agent_config(agent_id)
        if not agent_config:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent with ID '{agent_id}' not found.")
        
        executor, agent_mcp_client, discord_bot_id, telegram_bot_id = \
            await agent_manager.create_dynamic_agent_instance(agent_config, LOCAL_MODE)
        
        agent_manager.add_initialized_agent(
            agent_id, 
            agent_config.name, 
            executor, 
            agent_mcp_client,
            discord_bot_id=discord_bot_id,
            telegram_bot_id=telegram_bot_id 
        )
        agent_info = agent_manager.get_initialized_agent(agent_id)
    
    agent_executor: LangGraphADKAgent = agent_info["executor"]

    # For Streamlit chat, the agent's internal logic will use its own tools.

    logger.info(f"Invoking ADK agent '{agent_id}' with message: {user_message}")
    logger.info(f"DEBUG: User message being passed to agent.ainvoke: '{user_message}'")

    try:
        adk_input_message = Message(text=user_message)
        adk_output_message = await agent_executor.act(adk_input_message)

        final_message_content = adk_output_message.text if adk_output_message.text else "No response from agent."
        logger.info(f"Agent '{agent_id}' generated a final message: {final_message_content}")
        return {"response": final_message_content}
    except Exception as e:
        logger.error(f"Error chatting with agent '{agent_id}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error during agent interaction: {e}")


# --------- Internal A2A Invocation ---------
@app.post("/internal/agents/{agent_id}/invoke_adk")
async def invoke_adk_agent_internal(agent_id: str, adk_message: Message):
    """
    Internal endpoint for the ADK A2A server to invoke a specific agent.
    This endpoint should NOT be exposed externally.
    """
    logger.info(f"Received internal ADK invocation for agent '{agent_id}' with message: {adk_message.text[:100]}...")
    agent_info = agent_manager.get_initialized_agent(agent_id)
    if not agent_info:
        # Attempt to load from DB if not in cache
        agent_config = await app.state.db_manager.get_agent_config(agent_id)
        if not agent_config:
            logger.error(f"Internal ADK invocation: Agent with ID '{agent_id}' not found in cache or DB.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent with ID '{agent_id}' not found for internal invocation.")

        executor, agent_mcp_client, discord_bot_id, telegram_bot_id = \
            await agent_manager.create_dynamic_agent_instance(agent_config, LOCAL_MODE)

        agent_manager.add_initialized_agent(
            agent_id,
            agent_config.name,
            executor,
            agent_mcp_client,
            discord_bot_id=discord_bot_id,
            telegram_bot_id=telegram_bot_id
        )
        agent_info = agent_manager.get_initialized_agent(agent_id)

    agent_executor: LangGraphADKAgent = agent_info["executor"]

    try:
        adk_output_message = await agent_executor.act(adk_message)
        logger.info(f"Internal ADK invocation for agent '{agent_id}' completed. Response: {adk_output_message.text[:100]}...")
        return adk_output_message
    except Exception as e:
        logger.error(f"Error during internal ADK agent invocation for '{agent_id}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error during internal ADK agent invocation: {e}")


# --------- Telegram Webhook ----------

@app.post("/telegram/webhook")
async def tg_webhook(request: Request):
    try:
        data = await request.json()
        logger.info(f"Received Telegram webhook data: {json.dumps(data, indent=2)}")

        # Attempt to get message data, handling potential absence gracefully
        message = data.get("message")
        if not message:
            logger.debug("Webhook data does not contain a 'message' object. Assuming it's a forwarded payload.")
            chat_id = data.get("chat_id")
            user_id = data.get("user_id")
            user_message = data.get("content")
            message_id = data.get("message_id")
            user_name = data.get("user_name")
        else:
            # This handles raw Telegram webhooks (which should go to MCP first)
            logger.debug("Webhook data contains a 'message' object.")
            chat_id = message.get("chat", {}).get("id")
            user_id = message.get("from", {}).get("id")
            user_message = message.get("text")
            message_id = message.get("message_id")
            user_name = message.get("from", {}).get("username") or message.get("from", {}).get("first_name") or str(user_id)
        
        # The bot_id is expected to be at the top level for forwarded webhooks from MCP
        incoming_bot_id = data.get('bot_id') 

        if not all([chat_id, user_id, user_message, incoming_bot_id]):
            logger.warning(f"Missing essential Telegram message data. Skipping processing. Details: chat_id={chat_id}, user_id={user_id}, user_message={user_message}, bot_id={incoming_bot_id}")
            return {"status": "ok"}

        logger.info(f"Received Telegram message from user {user_id} in chat {chat_id} (via bot {incoming_bot_id}): {user_message}")

        # --- Agent Selection Logic for Telegram Webhook ---
        selected_agent_info = None
        logger.debug(f"Attempting to find agent for incoming_bot_id: {incoming_bot_id}")
        all_cached_telegram_ids = {name: info.get('telegram_bot_id') for name, info in agent_manager._initialized_agents.items()}
        logger.debug(f"Currently cached Telegram bot IDs: {all_cached_telegram_ids}")

        for agent_id, agent_info in agent_manager._initialized_agents.items():
            cached_telegram_bot_id = agent_info.get("telegram_bot_id")
            logger.debug(f"Checking agent '{agent_info.get('name')}' (ID: {agent_id}) with cached Telegram Bot ID: {cached_telegram_bot_id}. Comparing to incoming bot_id: {incoming_bot_id}")
            
            # DefaultBot should NOT reply to Telegram
            if agent_info["name"] == "DefaultBot":
                continue 
            
            # Check if this agent has the Telegram sending tool AND its configured bot_id matches the incoming bot_id
            if agent_info["mcp_client"].tools.get("send_message_telegram") and cached_telegram_bot_id == incoming_bot_id:
                selected_agent_info = agent_info
                logger.info(f"Selected agent '{agent_info['name']}' (ID: {agent_id}) for Telegram webhook based on bot ID match.")
                break # Found a suitable agent, use it

        if not selected_agent_info:
            logger.warning(f"No suitable agent found with Telegram API keys matching bot ID '{incoming_bot_id}' to reply to this message. Message ignored. Available agents' Telegram IDs: {[info.get('telegram_bot_id') for info in agent_manager._initialized_agents.values() if info.get('telegram_bot_id')]}")
            return {"status": "ignored", "detail": f"No agent configured for Telegram replies via bot ID {incoming_bot_id}."}
        
        agent_executor: LangGraphADKAgent = selected_agent_info["executor"]
        agent_mcp_client = selected_agent_info["mcp_client"]

        logger.info(f"Invoking agent '{selected_agent_info['name']}' with Telegram message...")

        adk_input_message = Message(
            text=user_message,
            metadata={"chat_id": str(chat_id), "user_id": str(user_id), "platform": "telegram"}
        )
        adk_output_message = await agent_executor.act(adk_input_message)

        final_message_content = adk_output_message.text if adk_output_message.text else "I'm sorry, I couldn't process that."

        logger.info(f"Agent '{selected_agent_info['name']}' generated Telegram reply: {final_message_content}")

        # Use the selected agent's MCP client for sending the reply
        telegram_tool = agent_mcp_client.tools.get("send_message_telegram")
        if telegram_tool:
            logger.info(f"Using agent '{selected_agent_info['name']}'s own Telegram tool to send reply.")
            send_result = await telegram_tool.ainvoke({ 
                "chat_id": str(chat_id),
                "message": final_message_content
            })
            logger.info(f"Telegram send_message tool call result: {send_result}")
        else:
            logger.error(f"Selected agent '{selected_agent_info['name']}' unexpectedly does not have 'send_message_telegram' tool. Cannot send reply.")

        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Error processing Telegram webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --------- Discord Webhook ---------

@app.post("/discord/receive_message")
async def receive_discord_message(payload: ReceiveDiscordMessageRequest):
    try:
        channel_id = payload.channel_id
        author_id = payload.author_id
        author_name = payload.author_name
        message_content = payload.content
        incoming_bot_id = payload.bot_id # Get the bot ID from the incoming payload

        logger.info(f"Received Discord message from {author_name} ({author_id}) via bot {incoming_bot_id} in channel {channel_id}: {message_content}")

        # --- Agent Selection Logic for Discord Webhook ---
        selected_agent_info = None
        # Iterate through all initialized agents to find the one configured for this Discord bot
        for agent_id, agent_info in agent_manager._initialized_agents.items():
            # DefaultBot should NOT reply to Discord messages
            if agent_info["name"] == "DefaultBot":
                continue 
            
            # Check if this agent has the Discord sending tool AND its configured bot_id matches the incoming bot_id
            agent_discord_bot_id = agent_info.get("discord_bot_id")
            
            if agent_info["mcp_client"].tools.get("send_message") and agent_discord_bot_id == incoming_bot_id:
                selected_agent_info = agent_info
                logger.info(f"Selected agent '{agent_info['name']}' (ID: {agent_id}) for Discord webhook based on bot ID match.")
                break # Found the specific agent, break loop

        if not selected_agent_info:
            logger.warning(f"No suitable agent found with Discord API keys matching bot ID '{incoming_bot_id}' to reply to this message. Message ignored.")
            return {"status": "ignored", "detail": f"No agent configured for Discord replies via bot ID {incoming_bot_id}."}
        
        agent_executor: LangGraphADKAgent = selected_agent_info["executor"]
        agent_mcp_client = selected_agent_info["mcp_client"]

        logger.info(f"Invoking agent '{selected_agent_info['name']}' with Discord message...")

        adk_input_message = Message(
            text=message_content,
            metadata={
                "channel_id": str(channel_id), 
                "author_id": str(author_id), 
                "platform": "discord"
                }
        )        

        adk_output_message = await agent_executor.act(adk_input_message)

        final_message_content = adk_output_message.text if adk_output_message.text else "I'm sorry, I couldn't process that."

        logger.info(f"Agent '{selected_agent_info['name']}' generated Discord reply: {final_message_content}")

        # Use the selected agent's MCP client for sending the reply
        discord_tool = agent_mcp_client.tools.get("send_message")
        if discord_tool:
            logger.info(f"Using agent '{selected_agent_info['name']}'s own Discord tool to send reply.")
            send_result = await discord_tool.ainvoke({
                "channel_id": str(channel_id),
                "message": final_message_content
            })
            logger.info(f"Discord send_message tool call result: {send_result}")
        else:
            logger.error(f"Selected agent '{selected_agent_info['name']}' unexpectedly does not have 'send_message' tool. Cannot send reply.")

        return {"status": "ok"}

    except ValidationError as e:
        logger.warning(f"Discord message validation failed: {e.errors()}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid Discord message payload: {e.errors()}")
    except Exception as e:
        logger.error(f"Error processing received Discord message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

