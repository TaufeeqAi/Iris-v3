import sqlite3
import json
import logging
import uuid
import asyncio # Import asyncio for running blocking operations in a thread
from typing import List, Optional, Any

from agent.models.agent_config import AgentConfig, AgentSecrets # Import Pydantic models
from pydantic import ValidationError # For error handling during parsing

logger = logging.getLogger(__name__)

class SQLiteManager:
    def __init__(self, database_file: str = "agents.db"):
        self.database_file = database_file
        # self.conn: Optional[sqlite3.Connection] = None # Removed: Connection will be created per-thread
        self._ensure_db_schema() # Ensure schema exists on init

    def _ensure_db_schema(self):
        """Ensures the SQLite database schema exists. This runs in the main thread."""
        conn = None
        try:
            conn = sqlite3.connect(self.database_file)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    bio TEXT,
                    knowledge TEXT,
                    persona TEXT,
                    secrets TEXT -- Stored as JSON string
                )
            ''')
            conn.commit()
            logger.info(f"SQLite database schema for '{self.database_file}' ensured.")
        except sqlite3.Error as e:
            logger.critical(f"Failed to ensure SQLite database schema: {e}", exc_info=True)
            raise
        finally:
            if conn:
                conn.close() # Close this connection immediately

    def close(self):
        """
        Closes the SQLite database connection.
        This method is now largely a no-op as connections are managed per-thread.
        Kept for API consistency.
        """
        logger.info(f"SQLiteManager close called. Connections are managed per-thread.")
        pass # Connections are now short-lived within asyncio.to_thread calls

    async def save_agent_config(self, agent_config: AgentConfig) -> str:
        """Saves or updates an agent configuration to SQLite asynchronously."""
        if agent_config.id is None:
            agent_config.id = str(uuid.uuid4()) # Generate a new ID if not provided

        # Run blocking SQLite operation in a separate thread
        def _save():
            conn = None
            try:
                conn = sqlite3.connect(self.database_file) # Connect in this thread
                cursor = conn.cursor()
                secrets_json = agent_config.secrets.model_dump_json(exclude_none=True)
                cursor.execute(
                    '''
                    INSERT OR REPLACE INTO agents (id, name, bio, knowledge, persona, secrets)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''',
                    (agent_config.id,
                     agent_config.name,
                     agent_config.bio,
                     agent_config.knowledge,
                     agent_config.persona,
                     secrets_json)
                )
                conn.commit()
                logger.info(f"Saved agent config '{agent_config.name}' with ID '{agent_config.id}' to SQLite.")
                return agent_config.id
            except sqlite3.Error as e:
                logger.error(f"SQLite error saving agent config: {e}", exc_info=True)
                raise
            finally:
                if conn:
                    conn.close() # Close connection after use
        
        return await asyncio.to_thread(_save)

    async def get_all_agent_configs(self) -> List[AgentConfig]:
        """Retrieves all agent configurations from SQLite asynchronously."""
        # Run blocking SQLite operation in a separate thread
        def _get_all():
            conn = None
            try:
                conn = sqlite3.connect(self.database_file) # Connect in this thread
                cursor = conn.cursor()
                cursor.execute("SELECT id, name, bio, knowledge, persona, secrets FROM agents")
                rows = cursor.fetchall()
                configs = []
                for row in rows:
                    try:
                        secrets_data = json.loads(row[5]) if row[5] else {}
                        secrets = AgentSecrets(**secrets_data)

                        config = AgentConfig(
                            id=row[0],
                            name=row[1],
                            bio=row[2],
                            knowledge=row[3],
                            persona=row[4],
                            secrets=secrets
                        )
                        configs.append(config)
                    except (json.JSONDecodeError, ValidationError) as e:
                        logger.error(f"Error parsing agent config from DB for ID {row[0]}: {e}")
                        continue
                logger.info(f"Retrieved {len(configs)} agent configs from SQLite.")
                return configs
            except sqlite3.Error as e:
                logger.error(f"SQLite error retrieving all agent configs: {e}", exc_info=True)
                raise
            finally:
                if conn:
                    conn.close() # Close connection after use
        
        return await asyncio.to_thread(_get_all)

    async def get_agent_config(self, agent_id: str) -> Optional[AgentConfig]:
        """Retrieves a specific agent configuration from SQLite by ID asynchronously."""
        # Run blocking SQLite operation in a separate thread
        def _get_by_id():
            conn = None
            try:
                conn = sqlite3.connect(self.database_file) # Connect in this thread
                cursor = conn.cursor()
                cursor.execute("SELECT id, name, bio, knowledge, persona, secrets FROM agents WHERE id = ?", (agent_id,))
                row = cursor.fetchone()
                if row:
                    try:
                        secrets_data = json.loads(row[5]) if row[5] else {}
                        secrets = AgentSecrets(**secrets_data)
                        return AgentConfig(
                            id=row[0],
                            name=row[1],
                            bio=row[2],
                            knowledge=row[3],
                            persona=row[4],
                            secrets=secrets
                        )
                    except (json.JSONDecodeError, ValidationError) as e:
                        logger.error(f"Error parsing agent config from DB for ID {agent_id}: {e}")
                        return None
                logger.warning(f"Agent config with ID '{agent_id}' not found in SQLite.")
                return None
            except sqlite3.Error as e:
                logger.error(f"SQLite error retrieving agent config by ID: {e}", exc_info=True)
                raise
            finally:
                if conn:
                    conn.close() # Close connection after use
            
        return await asyncio.to_thread(_get_by_id)

    async def delete_agent_config(self, agent_id: str) -> bool:
        """Deletes an agent configuration from SQLite by ID asynchronously."""
        # Run blocking SQLite operation in a separate thread
        def _delete():
            conn = None
            try:
                conn = sqlite3.connect(self.database_file) # Connect in this thread
                cursor = conn.cursor()
                cursor.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
                conn.commit()
                if cursor.rowcount > 0:
                    logger.info(f"Agent '{agent_id}' deleted from SQLite.")
                    return True
                else:
                    logger.warning(f"Agent '{agent_id}' not found for deletion in SQLite.")
                    return False
            except sqlite3.Error as e:
                logger.error(f"SQLite error deleting agent config: {e}", exc_info=True)
                raise
            finally:
                if conn:
                    conn.close() # Close connection after use
        
        return await asyncio.to_thread(_delete)

