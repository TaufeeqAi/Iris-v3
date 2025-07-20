# agent-UI/app.py

import streamlit as st
import requests
import json
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel # For AgentCard model

# --- Configuration ---
# Get BOT_API_BASE_URL from environment variable
BOT_API_BASE_URL = os.getenv("BOT_API_BASE_URL", "http://localhost:8000")

# --- Pydantic Models (Re-define or import if common/models.py is accessible) ---
# For simplicity, redefine AgentCard here. In a real multi-repo setup,
# you'd ideally share this model via a common Python package or direct import if paths allow.
class AgentSecrets(BaseModel):
    groq_api_key: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_api_id: Optional[int] = None
    telegram_api_hash: Optional[str] = None
    discord_bot_token: Optional[str] = None
    libp2p_private_key: Optional[str] = None # Added in Phase 1

class AgentConfig(BaseModel):
    id: Optional[str] = None # libp2p PeerID
    name: str
    bio: Optional[str] = None
    persona: Optional[str] = None
    secrets: AgentSecrets

class AgentCard(BaseModel):
    """
    Represents a discoverable agent in the decentralized marketplace.
    Matches the common/models.py AgentCard.
    """
    peer_id: str
    name: str
    bio: str
    capabilities: List[str]
    internal_url: str # Internal URL for ADK A2A server to invoke this agent


# --- API Client Functions ---

def create_agent_api(agent_data: Dict[str, Any]) -> requests.Response:
    """Sends a request to create a new agent."""
    url = f"{BOT_API_BASE_URL}/agents/create"
    return requests.post(url, json=agent_data)

def list_agents_api() -> List[Dict[str, Any]]:
    """Fetches all registered agents."""
    url = f"{BOT_API_BASE_URL}/agents/list"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error listing agents: {e}")
        return []

def chat_with_agent_api(agent_id: str, message: str) -> str:
    """Sends a message to a specific agent and returns its response."""
    url = f"{BOT_API_BASE_URL}/agents/{agent_id}/chat"
    try:
        response = requests.post(url, json={"message": message})
        response.raise_for_status()
        return response.json().get("response", "No response.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error chatting with agent: {e}")
        return f"Error: {e}"

def list_marketplace_agents_api(capabilities: Optional[List[str]] = None) -> List[AgentCard]:
    """
    Calls the bot-api to list agents from the decentralized marketplace.
    """
    url = f"{BOT_API_BASE_URL}/agents/marketplace/list"
    params = {}
    if capabilities:
        params["capabilities"] = ",".join(capabilities) # Pass as comma-separated string
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        # Parse the raw JSON list into AgentCard objects
        return [AgentCard(**agent_data) for agent_data in response.json()]
    except requests.exceptions.RequestException as e:
        st.error(f"Error listing agents from marketplace: {e}")
        return []

# --- Streamlit UI Components ---

def agent_management_page():
    st.header("🤖 Agent Management")

    st.subheader("Create New Agent")
    with st.form("create_agent_form"):
        agent_name = st.text_input("Agent Name", key="create_name")
        agent_bio = st.text_area("Agent Bio (Optional)", key="create_bio")
        agent_persona = st.text_area("Agent Persona (Optional)", key="create_persona")
        
        st.markdown("---")
        st.subheader("Agent Secrets (API Keys)")
        groq_api_key = st.text_input("Groq API Key", type="password", key="create_groq")
        
        st.markdown("#### Telegram Secrets (Optional)")
        telegram_bot_token = st.text_input("Telegram Bot Token", type="password", key="create_tg_token")
        telegram_api_id = st.number_input("Telegram API ID", min_value=0, step=1, key="create_tg_api_id")
        telegram_api_hash = st.text_input("Telegram API Hash", type="password", key="create_tg_api_hash")

        st.markdown("#### Discord Secrets (Optional)")
        discord_bot_token = st.text_input("Discord Bot Token", type="password", key="create_dc_token")

        submitted = st.form_submit_button("Create Agent")

        if submitted:
            secrets_data = AgentSecrets(
                groq_api_key=groq_api_key if groq_api_key else None,
                telegram_bot_token=telegram_bot_token if telegram_bot_token else None,
                telegram_api_id=telegram_api_id if telegram_api_id > 0 else None,
                telegram_api_hash=telegram_api_hash if telegram_api_hash else None,
                discord_bot_token=discord_bot_token if discord_bot_token else None,
            )
            agent_data = AgentConfig(
                name=agent_name,
                bio=agent_bio if agent_bio else None,
                persona=agent_persona if agent_persona else None,
                secrets=secrets_data
            )
            
            try:
                response = create_agent_api(agent_data.dict())
                response.raise_for_status()
                st.success(f"Agent '{agent_name}' created successfully! ID: {response.json().get('id')}")
                st.session_state.agents = None # Clear cache to refetch
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to create agent: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

    st.subheader("Existing Agents")
    if 'agents' not in st.session_state or st.session_state.agents is None:
        st.session_state.agents = list_agents_api()

    if st.session_state.agents:
        agent_names = {agent['id']: agent['name'] for agent in st.session_state.agents}
        selected_agent_id = st.selectbox("Select Agent to Chat With", options=list(agent_names.keys()), format_func=lambda x: agent_names[x], key="select_chat_agent")

        if selected_agent_id:
            st.session_state.selected_agent_id = selected_agent_id
            st.session_state.selected_agent_name = agent_names[selected_agent_id]
            st.write(f"Chatting with: **{st.session_state.selected_agent_name}** (ID: `{st.session_state.selected_agent_id}`)")

            # Initialize chat history for the selected agent
            if f"messages_{selected_agent_id}" not in st.session_state:
                st.session_state[f"messages_{selected_agent_id}"] = []

            # Display chat messages
            for message in st.session_state[f"messages_{selected_agent_id}"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Say something..."):
                st.session_state[f"messages_{selected_agent_id}"].append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.spinner("Agent thinking..."):
                    agent_response = chat_with_agent_api(selected_agent_id, prompt)
                
                st.session_state[f"messages_{selected_agent_id}"].append({"role": "assistant", "content": agent_response})
                with st.chat_message("assistant"):
                    st.markdown(agent_response)
    else:
        st.info("No agents created yet. Use the form above to create one.")

def agent_marketplace_page():
    st.header("🌐 Agent Marketplace")
    st.write("Discover and connect with agents in the decentralized network.")

    # Filter by capabilities
    available_capabilities = ["chat", "general_query", "finance_queries", "knowledge_retrieval", "telegram_bot", "discord_bot"]
    selected_capabilities = st.multiselect(
        "Filter by Capabilities",
        options=available_capabilities,
        key="marketplace_capabilities"
    )

    if st.button("List Agents from Marketplace", key="list_marketplace_agents_button"):
        with st.spinner("Searching marketplace..."):
            st.session_state.marketplace_agents = list_marketplace_agents_api(selected_capabilities)
    
    if 'marketplace_agents' in st.session_state and st.session_state.marketplace_agents:
        st.subheader("Discovered Agents")
        for agent_card in st.session_state.marketplace_agents:
            with st.expander(f"**{agent_card.name}** (ID: `{agent_card.peer_id[:10]}...`)"):
                st.write(f"**Bio:** {agent_card.bio}")
                st.write(f"**Capabilities:** {', '.join(agent_card.capabilities)}")
                
                # "Connect with Agent" Button
                if st.button(f"Connect with {agent_card.name}", key=f"connect_{agent_card.peer_id}"):
                    st.session_state.selected_agent_id = agent_card.peer_id
                    st.session_state.selected_agent_name = agent_card.name
                    st.session_state.page = "Agent Management" # Switch back to chat page
                    st.success(f"Switched to chat with {agent_card.name}. Go to 'Agent Management' tab.")
                    st.experimental_rerun() # Rerun to switch page

    else:
        st.info("No agents discovered in the marketplace yet, or try different filters.")

    st.subheader("Share Your Agent (Conceptual)")
    st.write("This feature will allow you to make your created agents discoverable in the decentralized marketplace.")
    st.write("*(Implementation Note: This would involve calling a backend endpoint to publish the agent's PeerID and capabilities to the libp2p DHT via the agent-marketplace-mcp.)*")
    
    # Placeholder for share agent functionality
    if 'agents' in st.session_state and st.session_state.agents:
        shareable_agent_ids = {agent['id']: agent['name'] for agent in st.session_state.agents}
        selected_agent_to_share = st.selectbox(
            "Select your agent to share to the marketplace",
            options=list(shareable_agent_ids.keys()),
            format_func=lambda x: shareable_agent_ids[x],
            key="select_agent_to_share"
        )
        if st.button(f"Share '{shareable_agent_ids.get(selected_agent_to_share, '')}'", key="share_agent_button"):
            st.info(f"Agent '{shareable_agent_ids[selected_agent_to_share]}' (ID: {selected_agent_to_share}) is conceptually shared. Backend integration needed to publish to DHT.")
            # In Phase 4, you'd call a new API endpoint here:
            # requests.post(f"{BOT_API_BASE_URL}/agents/{selected_agent_to_share}/share_to_marketplace")


# --- Main App Logic ---
st.set_page_config(layout="wide", page_title="Multi-Agent Bot")
st.title("Multi-Agent Bot System")

# Page selection using radio buttons or sidebar
if 'page' not in st.session_state:
    st.session_state.page = "Agent Management"

page_selection = st.sidebar.radio("Navigation", ["Agent Management", "Agent Marketplace"])

if page_selection == "Agent Management":
    agent_management_page()
elif page_selection == "Agent Marketplace":
    agent_marketplace_page()

