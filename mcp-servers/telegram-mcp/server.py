import os
from fastapi import FastAPI, Request, HTTPException
from fastmcp import FastMCP
from telethon import TelegramClient, events
from telethon.tl.types import User as TelegramUser
import logging
import json
from contextlib import asynccontextmanager
import httpx
from dotenv import load_dotenv
import asyncio

load_dotenv()

logger = logging.getLogger(__name__)
try:
    from common.utils import setup_logging
    setup_logging(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.warning("common.utils.setup_logging not found, using basic logging config.")

logging.getLogger('telethon').setLevel(logging.INFO)
logging.getLogger('FastMCP').setLevel(logging.INFO) 
logging.getLogger('uvicorn').setLevel(logging.INFO) 


BOT_API_BASE_URL = os.getenv("BOT_API_BASE_URL", "http://localhost:8000")

# --- TelegramClientManager ---
class TelegramClientManager:
    def __init__(self):
        self.clients = {}
        self.bot_token_to_id = {}
        self.id_to_bot_token = {}
        self.lock = asyncio.Lock()

    async def get_or_create_client(self, bot_token: str, api_id: int, api_hash: str) -> TelegramClient:
        async with self.lock:
            # Check if client already exists for this token
            if bot_token in self.bot_token_to_id:
                bot_id = self.bot_token_to_id[bot_token]
                client = self.clients.get(bot_id)
                if client and client.is_connected():
                    logger.debug(f"Re-using existing TelegramClient for bot_id: {bot_id}")
                    return client
                elif client:
                    logger.warning(f"Existing client for bot_id {bot_id} found but not connected. Attempting restart/recreate.")
                    await client.disconnect()
                    del self.clients[bot_id]
                    del self.bot_token_to_id[bot_token]
                    del self.id_to_bot_token[bot_id]
            
            logger.info(f"Creating new TelegramClient for bot_token: {bot_token[:5]}... (API ID: {api_id})")
            session_name = f'bot_session_{bot_token[:8]}' # Unique session name per token
            client = TelegramClient(session_name, api_id, api_hash)
            
            try:
                await client.start(bot_token=bot_token)
                if not client.is_connected():
                    raise ConnectionError("Telethon client could not establish an active connection.")

                bot_info: TelegramUser = await client.get_me()
                if not bot_info:
                    raise ConnectionError("Telethon client connected but failed to retrieve bot info via get_me().")
                
                client._bot_info = bot_info 
                
                bot_id = str(bot_info.id)
                
                self.clients[bot_id] = client
                self.bot_token_to_id[bot_token] = bot_id
                self.id_to_bot_token[bot_id] = bot_id
                
                client.add_event_handler(lambda e: handle_telegram_message(e, client, self), events.NewMessage(incoming=True))
                logger.info(f"New TelegramClient created, connected, and handler registered for Bot ID: {bot_id}")
                return client
            except Exception as e:
                logger.error(f"Error starting Telethon client for token {bot_token[:5]}...: {e}", exc_info=True)
                if client.is_connected():
                    await client.disconnect()
                raise HTTPException(status_code=500, detail=f"Failed to connect Telegram bot client: {e}")

    async def get_client_by_bot_id(self, bot_id: str) -> TelegramClient:
        async with self.lock:
            client = self.clients.get(bot_id)
            if not client or not client.is_connected():
                logger.error(f"Client for bot ID {bot_id} not found or not connected.")
                raise HTTPException(status_code=500, detail=f"Telegram client for bot ID {bot_id} is not active.")
            return client

    async def shutdown_all_clients(self):
        async with self.lock:
            for bot_id, client in list(self.clients.items()): # Iterate on a copy
                if client.is_connected():
                    logger.info(f"Disconnecting TelegramClient for bot ID: {bot_id} during shutdown.")
                    await client.disconnect()
                del self.clients[bot_id]
                logger.info(f"Client for bot ID {bot_id} removed from manager.")
            self.clients = {} # Clear the dictionary
            self.bot_token_to_id = {}
            self.id_to_bot_token = {}
            logger.info("All Telegram clients disconnected and manager cleared.")


telegram_client_manager = TelegramClientManager()

# Create the FastMCP instance
mcp = FastMCP("telegram") # Unique name for this MCP

http_mcp = mcp.http_app(transport="streamable-http")

#combined lifespan context manager that wraps FastMCP's lifespan
@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    logger.info("Application lifespan startup initiated for Telegram MCP.")
    # Enter FastMCP's lifespan context
    async with http_mcp.router.lifespan_context(app) as maybe_state:
        # Yield control to the application during startup
        yield maybe_state
    
    logger.info("Application lifespan shutdown initiated. Disconnecting all managed Telethon clients...")
    await telegram_client_manager.shutdown_all_clients()


# Create the main FastAPI app and pass the combined lifespan manager
app = FastAPI(lifespan=combined_lifespan)

# Mount http_mcp's router onto the main app
app.mount("/", http_mcp)


async def handle_telegram_message(event, client: TelegramClient, manager: TelegramClientManager):
    message = event.message
    if message.out:
        logger.debug(f"Ignoring outgoing message from bot: {message.text[:50]}...")
        return
    if message.is_channel or (message.is_group and not message.is_private):
        logger.debug(f"Ignoring group/channel message (for now): {message.text[:50]}...")
        pass

    sender = await message.get_sender()
    sender_name = sender.username or sender.first_name or str(sender.id)
    chat_id = str(message.chat_id)
    user_id = str(sender.id)
    user_message = message.text

    # Get the bot_id from the client that received the message
    current_bot_id = str(client._bot_info.id) if hasattr(client, '_bot_info') and client._bot_info else "UNKNOWN_BOT_ID"
    logger.info(f"Telegram received message from {sender_name} ({user_id}) in chat {chat_id} via bot {current_bot_id}: {user_message[:100]}...")

    msg_data = {
        "content": user_message,
        "chat_id": chat_id,
        "user_id": user_id,
        "user_name": sender_name,
        "message_id": str(message.id),
        "timestamp": message.date.isoformat(),
        "bot_id": current_bot_id
    }
    logger.debug(f"Forwarding payload to main API: {json.dumps(msg_data, indent=2)}")

    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                f"{BOT_API_BASE_URL}/telegram/webhook",
                json=msg_data,
                timeout=30.0
            )
            response.raise_for_status()
            logger.info(f"Successfully forwarded Telegram message to bot API. Response: {response.status_code}")
    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to forward Telegram message to bot API (HTTP error): {e.response.status_code} - {e.response.text}", exc_info=True)
    except httpx.RequestError as e:
        logger.error(f"Failed to forward Telegram message to bot API (Request error): {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred while forwarding Telegram message: {e}", exc_info=True)


@mcp.tool()
async def send_message_telegram(chat_id: str, message: str, telegram_bot_token: str, telegram_api_id: int, telegram_api_hash: str) -> str:
    """
    Sends a text message to a specific Telegram chat, group, or channel using a dynamically provided bot token.
    Use this tool when the user explicitly asks to send a message to a Telegram contact or group.

    :param chat_id: The unique identifier (ID) or username (e.g., "@my_channel", "user_id_number") of the Telegram chat, user, or channel to send the message to.
    :param message: The text content of the message to be sent.
    :param telegram_bot_token: The API token for the Telegram bot to use.
    :param telegram_api_id: The API ID for the Telegram bot's application.
    :param telegram_api_hash: The API Hash for the Telegram bot's application.
    :returns: A confirmation message indicating success or an error if the message could not be sent.
    """
    logger.info(f"Attempting to send message to Telegram chat {chat_id} using token {telegram_bot_token[:5]}...: {message[:50]}...")
    
    try:
        client = await telegram_client_manager.get_or_create_client(telegram_bot_token, telegram_api_id, telegram_api_hash)
        
        logger.debug(f"Telethon client connected status before send: {client.is_connected()}")
        bot_info_for_send = await client.get_me()
        current_bot_id_for_send = str(bot_info_for_send.id) if bot_info_for_send else "UNKNOWN_BOT_ID_FROM_GET_ME"
        logger.debug(f"Telethon client's current bot ID for send (from get_me()): {current_bot_id_for_send}")

        target_entity = None
        try:
            target_entity = int(chat_id)
            logger.debug(f"Directly using integer chat_id for send_message: {target_entity}")
        except ValueError:
            logger.debug(f"chat_id '{chat_id}' is not an integer. Attempting to get entity using get_entity.")
            target_entity = await client.get_entity(chat_id)
            logger.debug(f"Entity obtained for '{chat_id}': {target_entity.id} (username/title: {target_entity.username or target_entity.title or 'N/A'})")
        
        logger.info(f"Sending message via Telethon to target_entity: {target_entity} using bot ID: {current_bot_id_for_send}...")
        sent_message = await client.send_message(target_entity, message) 
        logger.info(f"Message successfully sent to {chat_id} by bot {current_bot_id_for_send}. Message ID: {sent_message.id}")
        return f"Message successfully sent to {chat_id}."
    except Exception as e:
        logger.error(f"Error sending message to Telegram chat {chat_id} using token {telegram_bot_token[:5]}...: {e}", exc_info=True) 
        return f"Error sending message: {e}"

@mcp.tool()
async def get_chat_history(chat_id: str, telegram_bot_token: str, telegram_api_id: int, telegram_api_hash: str, limit: int = 10) -> str:
    """
    Retrieves a specified number of recent messages from a Telegram chat, group, or channel's history
    using a dynamically provided bot token.

    :param chat_id: The unique identifier (ID) or username (e.g., "@my_channel", "user_id_number") of the Telegram chat, user, or channel from which to retrieve history.
    :param telegram_bot_token: The API token for the Telegram bot to use.
    :param telegram_api_id: The API ID for the Telegram bot's application.
    :param telegram_api_hash: The API Hash for the Telegram bot's application.
    :param limit: The maximum number of most recent messages to retrieve (default is 10).
    :returns: A JSON string containing a list of recent messages, including sender, text, and date.
    """
    logger.info(f"Attempting to retrieve chat history for Telegram chat {chat_id} (limit: {limit}) using token {telegram_bot_token[:5]}...")
    
    try:
        client = await telegram_client_manager.get_or_create_client(telegram_bot_token, telegram_api_id, telegram_api_hash)

        logger.debug(f"Telethon client connected status before history fetch: {client.is_connected()}")
        # Call get_me() again for the most up-to-date bot info for this specific operation
        bot_info_for_history = await client.get_me()
        current_bot_id_for_history = str(bot_info_for_history.id) if bot_info_for_history else "UNKNOWN_BOT_ID_FROM_GET_ME"
        logger.debug(f"Telethon client's current bot ID for history (from get_me()): {current_bot_id_for_history}")

        target_entity = None
        try:
            target_entity = int(chat_id)
            logger.debug(f"Directly using integer chat_id for history: {target_entity}")
        except ValueError:
            target_entity = await client.get_entity(chat_id)
            logger.debug(f"Entity obtained for history: {target_entity.id}")

        messages_list = []
        async for msg in client.iter_messages(target_entity, limit=limit): 
            messages_list.append({
                "id": str(msg.id),
                "sender": msg.sender.username or msg.sender.first_name or "Unknown",
                "text": msg.text,
                "date": msg.date.isoformat() if msg.date else None
            })
        logger.info(f"Retrieved {len(messages_list)} messages from {chat_id} using bot ID: {current_bot_id_for_history}.")
        return json.dumps(messages_list, indent=2)
    except Exception as e:
        logger.error(f"Error retrieving chat history for Telegram chat {chat_id} using token {telegram_bot_token[:5]}...: {e}", exc_info=True)
        return f"Error retrieving chat history: {e}"

@mcp.tool()
async def get_bot_id_telegram(telegram_bot_token: str, telegram_api_id: int, telegram_api_hash: str) -> str:
    """
    Returns the Telegram bot's user ID for a dynamically provided bot token.
    This is useful for identifying which specific bot instance is running.

    :param telegram_bot_token: The API token for the Telegram bot to use.
    :param telegram_api_id: The API ID for the Telegram bot's application.
    :param telegram_api_hash: The API Hash for the Telegram bot's application.
    :returns: The Telegram bot's user ID.
    """
    logger.info(f"Attempting to retrieve bot ID for token {telegram_bot_token[:5]}...")
    try:
        client = await telegram_client_manager.get_or_create_client(telegram_bot_token, telegram_api_id, telegram_api_hash)
        
        bot_info = await client.get_me()
        
        if bot_info:
            bot_id = str(bot_info.id)
            logger.info(f"Providing Telegram bot ID for token {telegram_bot_token[:5]}...: {bot_id}")
            return bot_id
        else:
            logger.warning(f"get_me() returned None for token {telegram_bot_token[:5]}... Returning UNKNOWN_BOT_ID.")
            return "UNKNOWN_BOT_ID" 
    except Exception as e:
        logger.error(f"Failed to fetch Telegram Bot ID for token {telegram_bot_token[:5]}...: {e}", exc_info=True)
        raise ValueError(f"Telegram bot client not ready to provide ID for the provided token: {e}")

logger.info("Telegram MCP server initialized for dynamic client management.")
