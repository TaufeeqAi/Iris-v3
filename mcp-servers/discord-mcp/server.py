import os
import sys
import asyncio
import logging
import json
import httpx
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI
from fastmcp import FastMCP
import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

# Logging setup
logger = logging.getLogger(__name__)
try:
    from common.utils import setup_logging
    setup_logging(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.warning("common.utils.setup_logging not found, using basic logging config.")

mcp = FastMCP("discord")

BOT_API_BASE_URL = os.getenv("BOT_API_BASE_URL", "http://localhost:8000") 

_active_discord_bots: Dict[str, commands.Bot] = {}
_discord_bot_tasks: Dict[str, asyncio.Task] = {}

async def _start_discord_client(bot_token: str) -> Tuple[str, commands.Bot]:
    """
    Initializes and starts a new Discord bot client in a background task.
    Returns the bot's ID once it's ready.
    """
    intents = discord.Intents.default()
    intents.message_content = True 
    intents.members = True      


    bot_client = commands.Bot(command_prefix="!", intents=intents)

    ready_event = asyncio.Event()
    bot_id: Optional[str] = None

    @bot_client.event
    async def on_ready():
        nonlocal bot_id
        bot_id = str(bot_client.user.id)
        _active_discord_bots[bot_id] = bot_client 
        logger.info(f'Discord bot client logged in as {bot_client.user} (ID: {bot_id})')
        logger.info('Discord client ready!')
        ready_event.set() 

    @bot_client.event
    async def on_message(message: discord.Message):
        """
        Handles incoming Discord messages for this specific bot client.
        Forwards user messages to the main bot API for agent processing.
        """
        if message.author == bot_client.user:
            return

        if message.author.bot:
            return

        current_bot_id = str(bot_client.user.id)
        logger.info(f"Discord WebSocket received message for bot {current_bot_id} from {message.author.display_name} ({message.author.id}) in channel {message.channel.id}: {message.content[:100]}...")

        # Prepare message data to send to the main bot API
        msg_data = {
            "content": message.content,
            "channel_id": str(message.channel.id),
            "author_id": str(message.author.id),
            "author_name": message.author.display_name,
            "message_id": str(message.id),
            "timestamp": message.created_at.isoformat(),
            "bot_id": current_bot_id 
        }

        if message.guild:
            msg_data["guild_id"] = str(message.guild.id)
        else:
            msg_data["guild_id"] = None

        # Send the message data to the main bot API's /discord/receive_message endpoint
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BOT_API_BASE_URL}/discord/receive_message",
                    json=msg_data,
                    timeout=30.0
                )
                response.raise_for_status()
                logger.info(f"Successfully forwarded message to bot API. Response: {response.status_code}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to forward Discord message to bot API (HTTP error): {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Failed to forward Discord message to bot API (Request error): {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while forwarding Discord message: {e}")

    # Start the bot client in a background task
    task = asyncio.create_task(bot_client.start(bot_token))
    _discord_bot_tasks[bot_token] = task # Store the task for potential cancellation

    # Wait for the bot to be ready and its ID to be set
    await ready_event.wait()
    
    if bot_id is None:
        raise RuntimeError(f"Discord bot client for token (first 5 chars) {bot_token[:5]}... failed to get ID.")

    return bot_id, bot_client

@mcp.tool()
async def register_discord_bot(bot_token: str) -> str:
    """
    Registers and starts a new Discord bot client with the provided token.
    This tool should be called by the agent manager when an agent requires Discord capabilities.
    Returns the Discord bot's user ID once it's successfully logged in.
    """
    logger.info(f"Attempting to register and start Discord bot with token (first 5 chars): {bot_token[:5]}...")
    try:

        for existing_bot_id, existing_bot_client in _active_discord_bots.items():
            if existing_bot_client.http.token == bot_token:
                logger.info(f"Discord bot with ID {existing_bot_id} already registered and running for this token.")
                return existing_bot_id
        
        # If not already running, start a new client
        bot_id, _ = await _start_discord_client(bot_token)
        logger.info(f"Successfully registered and started Discord bot with ID: {bot_id}")
        return bot_id
    except Exception as e:
        logger.error(f"Error registering or starting Discord bot for token (first 5 chars) {bot_token[:5]}...: {e}", exc_info=True)
        raise

@mcp.tool()
async def send_message(bot_id: str, channel_id: str, message: str) -> str:
    """
    Sends a text message to a specific Discord channel using the specified bot.
    Use this tool when the user explicitly asks to send a message to a Discord channel or user.
    The bot must have permission to send messages in the specified channel.

    :param bot_id: The unique identifier (ID) of the Discord bot to use for sending the message.
    :param channel_id: The unique identifier (ID) of the Discord channel where the message should be sent.
    :param message: The text content of the message to send.
    :returns: A confirmation message indicating success or an error if the message could not be sent.
    """
    logger.info(f"Bot {bot_id}: Attempting to send message to Discord channel {channel_id}: {message[:50]}...")
    try:
        bot_client = _active_discord_bots.get(bot_id)
        if not bot_client or not bot_client.is_ready():
            raise ValueError(f"Discord bot with ID {bot_id} is not registered or not ready.")

        channel = bot_client.get_channel(int(channel_id))
        if not channel:
            channel = await bot_client.fetch_channel(int(channel_id))
        if not channel:
            raise ValueError(f"Discord channel with ID {channel_id} not found or inaccessible for bot {bot_id}.")

        await channel.send(message)
        logger.info(f"Bot {bot_id}: Message successfully sent to Discord channel {channel_id}.")
        return f"Message successfully sent by bot {bot_id} to Discord channel {channel_id}."
    except Exception as e:
        logger.error(f"Bot {bot_id}: Error sending message to Discord channel {channel_id}: {e}", exc_info=True)
        return f"Error sending message by bot {bot_id}: {e}"

@mcp.tool()
async def get_channel_messages(bot_id: str, channel_id: str, limit: int = 10) -> str:
    """
    Retrieves the recent message history from a specified Discord channel using the specified bot.
    This tool is useful for fetching context from a Discord conversation or reviewing past messages.
    The bot must have permission to read message history in the specified channel.

    :param bot_id: The unique identifier (ID) of the Discord bot to use for retrieving messages.
    :param channel_id: The unique identifier (ID) of the Discord channel from which to retrieve messages.
    :param limit: The maximum number of most recent messages to retrieve (default is 10).
    :returns: A JSON string containing a list of recent messages, including sender, content, and timestamp.
    """
    logger.info(f"Bot {bot_id}: Attempting to retrieve messages from Discord channel {channel_id} (limit: {limit}).")
    try:
        bot_client = _active_discord_bots.get(bot_id)
        if not bot_client or not bot_client.is_ready():
            raise ValueError(f"Discord bot with ID {bot_id} is not registered or not ready.")

        channel = bot_client.get_channel(int(channel_id))
        if not channel:
            channel = await bot_client.fetch_channel(int(channel_id))
        if not channel:
            raise ValueError(f"Discord channel with ID {channel_id} not found or inaccessible for bot {bot_id}.")

        messages_list = []
        async for msg in channel.history(limit=limit):
            messages_list.append({
                "id": str(msg.id),
                "author": msg.author.display_name if msg.author else "Unknown",
                "content": msg.content,
                "timestamp": msg.created_at.isoformat(),
                "channel_name": channel.name,
                "guild_name": channel.guild.name if channel.guild else "Direct Message"
            })
        logger.info(f"Bot {bot_id}: Retrieved {len(messages_list)} messages from Discord channel {channel_id}.")
        return json.dumps(messages_list, indent=2)
    except Exception as e:
        logger.error(f"Bot {bot_id}: Error retrieving messages from Discord channel {channel_id}: {e}", exc_info=True)
        return f"Error retrieving messages by bot {bot_id}: {e}"

@mcp.tool()
async def get_bot_id(bot_token: str) -> str:
    """
    Returns the Discord bot's user ID for a given bot token.
    This is useful for identifying which specific bot instance is running.
    This tool does not require the bot to be fully logged in, but it does
    a quick check to get the bot's ID from Discord's API.
    """
    logger.info(f"Attempting to get Discord bot ID for token (first 5 chars): {bot_token[:5]}...")
    try:

        temp_client = discord.Client(intents=discord.Intents.none())
        
        await temp_client.login(bot_token)
        
        bot_user = temp_client.user
        if bot_user:
            logger.info(f"Successfully fetched Discord bot ID: {bot_user.id} for token (first 5 chars): {bot_token[:5]}...")
            return str(bot_user.id)
        else:
            raise ValueError("Could not retrieve bot user information from token.")
    except Exception as e:
        logger.error(f"Error fetching Discord bot ID for token (first 5 chars) {bot_token[:5]}...: {e}", exc_info=True)
        raise ValueError(f"Failed to get Discord bot ID: {e}")

http_mcp = mcp.http_app(transport="streamable-http")

@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for the Discord MCP server.
    It manages the FastMCP application's lifecycle. Individual Discord bot clients
    are managed by the `register_discord_bot` tool.
    """
    logger.info("Starting FastMCP application lifecycle...")
    async with http_mcp.router.lifespan_context(app) as fastmcp_lifespan_yield:
        yield fastmcp_lifespan_yield # Yield control to the FastAPI application

    logger.info("Shutting down Discord MCP server. Cancelling all active Discord bot tasks...")
    for bot_token, task in _discord_bot_tasks.items():
        if not task.done():
            task.cancel()
            try:
                await task # Await cancellation to ensure cleanup
                logger.info(f"Discord bot task for token (first 5 chars) {bot_token[:5]}... cancelled and awaited.")
            except asyncio.CancelledError:
                logger.info(f"Discord bot task for token (first 5 chars) {bot_token[:5]}... was already cancelled.")
            except Exception as e:
                logger.error(f"Error during cancellation of Discord bot task for token (first 5 chars) {bot_token[:5]}...: {e}", exc_info=True)
    
    # Close all active bot clients
    for bot_id, bot_client in _active_discord_bots.items():
        if not bot_client.is_closed():
            await bot_client.close()
            logger.info(f"Discord bot client {bot_id} closed.")
    
    _active_discord_bots.clear()
    _discord_bot_tasks.clear()
    logger.info("All Discord bot clients and tasks cleaned up.")


app = FastAPI(lifespan=combined_lifespan)
app.mount("/", http_mcp)
logger.info("Discord MCP server initialized and tools registered.")

