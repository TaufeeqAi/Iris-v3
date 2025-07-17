import streamlit as st
import requests
import json
import os
from typing import Optional # <--- ADD THIS IMPORT

# --- Configuration ---
# Ensure your FastAPI backend is running on this URL
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Multi-Agent Bot Console",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
# This dictionary stores variables that persist across reruns of the Streamlit app
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'create_agent' # Default page
if 'selected_agent_id' not in st.session_state:
    st.session_state.selected_agent_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {} # {agent_id: [{"role": "user/assistant", "content": "message"}]}
if 'agents' not in st.session_state:
    st.session_state.agents = [] # Cached list of agents

# --- Helper Functions for FastAPI Communication ---

def send_request_to_fastapi(method: str, endpoint: str, data: Optional[dict] = None):
    """Sends an HTTP request to the FastAPI backend."""
    url = f"{FASTAPI_URL}/{endpoint}"
    try:
        if method.lower() == 'post':
            response = requests.post(url, json=data)
        elif method.lower() == 'get':
            response = requests.get(url)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return None

        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to FastAPI backend at {FASTAPI_URL}. Please ensure it is running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"FastAPI error: {e.response.status_code} - {e.response.text}")
        return None
    except json.JSONDecodeError:
        st.error(f"Failed to decode JSON response from FastAPI: {response.text}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def get_agents():
    """Fetches list of agents from FastAPI backend."""
    response = send_request_to_fastapi('get', 'agents/list')
    if response:
        st.session_state.agents = response # Update cached agents
    return st.session_state.agents

def create_agent_on_backend(agent_data: dict):
    """Sends new agent data to FastAPI backend to create an agent."""
    response = send_request_to_fastapi('post', 'agents/create', agent_data)
    if response:
        st.success(f"Agent '{response.get('name', 'Unknown')}' created successfully!")
        # Refresh the list of agents in the sidebar
        st.session_state.agents = get_agents()
        st.session_state.current_page = 'list_agents' # Navigate to list after creation
        st.rerun() # Rerun to update sidebar
    return response

def chat_with_agent_on_backend(agent_id: str, message: str):
    """Sends a chat message to a specific agent and gets a response."""
    endpoint = f"agents/{agent_id}/chat"
    data = {"message": message}
    response = send_request_to_fastapi('post', endpoint, data)
    if response:
        return response.get("response", "No response from agent.")
    return "Error: Could not get response from agent."

# --- UI Components ---

def create_agent_page():
    """Renders the 'Create New Agent' form."""
    st.title("➕ Create New Agent")
    st.write("Upload a JSON configuration file or fill the form to create a new AI agent.")

    # JSON Upload Option
    uploaded_file = st.file_uploader("Upload character.json file", type="json")
    if uploaded_file is not None:
        try:
            file_content = json.load(uploaded_file)
            st.session_state.uploaded_agent_config = file_content
            st.success("JSON file uploaded successfully! Review details below.")
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid JSON.")
            st.session_state.uploaded_agent_config = None

    st.subheader("Agent Details")
    with st.form("create_agent_form"):
        # Pre-fill from uploaded file if available
        default_name = st.session_state.uploaded_agent_config.get('name', '') if 'uploaded_agent_config' in st.session_state else ''
        default_bio = st.session_state.uploaded_agent_config.get('bio', '') if 'uploaded_agent_config' in st.session_state else ''
        default_knowledge = st.session_state.uploaded_agent_config.get('knowledge', '') if 'uploaded_agent_config' in st.session_state else ''
        default_persona = st.session_state.uploaded_agent_config.get('persona', '') if 'uploaded_agent_config' in st.session_state else ''
        default_secrets = json.dumps(st.session_state.uploaded_agent_config.get('secrets', {}), indent=2) if 'uploaded_agent_config' in st.session_state else json.dumps({
            "discord_bot_token": "YOUR_DISCORD_BOT_TOKEN",
            "telegram_api_id": "YOUR_TELEGRAM_API_ID",
            "telegram_api_hash": "YOUR_TELEGRAM_API_HASH",
            "telegram_bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
            "serpapi_api_key": "YOUR_SERPAPI_API_KEY",
            "newsapi_org_api_key": "YOUR_NEWSAPI_ORG_API_KEY",
            "finnhub_api_key": "YOUR_FINNHUB_API_KEY",
            "quandl_api_key": "YOUR_QUANDL_API_KEY",
            "cohere_api_key": "YOUR_COHERE_API_KEY",
            "groq_api_key": "YOUR_GROQ_API_KEY" # Optional, can use global
        }, indent=2)

        name = st.text_input("Agent Name", value=default_name, help="A unique name for your agent.")
        bio = st.text_area("Bio", value=default_bio, help="A brief description of your agent's background or purpose.")
        knowledge = st.text_area("Knowledge Areas", value=default_knowledge, help="Specific domains or topics your agent is knowledgeable about.")
        persona = st.text_area("Persona", value=default_persona, help="How your agent should behave (e.g., 'friendly', 'formal', 'humorous').")
        secrets_json_str = st.text_area("Secrets (JSON)", value=default_secrets, height=250, help="API keys for tools. Leave blank or remove keys for tools you don't want.")

        submitted = st.form_submit_button("Create Agent")

        if submitted:
            try:
                secrets_dict = json.loads(secrets_json_str)
                agent_data = {
                    "name": name,
                    "bio": bio,
                    "knowledge": knowledge,
                    "persona": persona,
                    "secrets": secrets_dict
                }
                create_agent_on_backend(agent_data)
            except json.JSONDecodeError:
                st.error("Invalid JSON in Secrets field. Please correct it.")
            except Exception as e:
                st.error(f"An error occurred during agent creation: {e}")

def chat_page(agent_id: str):
    """Renders the chat interface for a selected agent."""
    agents = get_agents() # Refresh agents to get current names
    agent_name = next((a['name'] for a in agents if a['id'] == agent_id), "Unknown Agent")
    st.title(f"💬 Chat with {agent_name}")

    if agent_id not in st.session_state.chat_history:
        st.session_state.chat_history[agent_id] = []

    # Display chat messages
    for message in st.session_state.chat_history[agent_id]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Say something to your agent..."):
        # Add user message to history
        st.session_state.chat_history[agent_id].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.spinner("Agent is thinking..."):
            agent_response = chat_with_agent_on_backend(agent_id, prompt)

        # Add agent response to history
        st.session_state.chat_history[agent_id].append({"role": "assistant", "content": agent_response})
        with st.chat_message("assistant"):
            st.markdown(agent_response)

def list_agents_page():
    """Renders the list of available agents."""
    st.title("📋 Available Agents")
    agents = get_agents() # Fetch latest agents

    if not agents:
        st.info("No agents created yet. Use the 'Create New Agent' option in the sidebar.")
    else:
        for agent in agents:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(agent.get('name', 'N/A'))
                st.write(f"ID: `{agent.get('id', 'N/A')}`")
                st.write(f"Bio: {agent.get('bio', 'No bio provided.')}")
                st.write(f"Persona: {agent.get('persona', 'No persona provided.')}")
            with col2:
                if st.button(f"Chat with {agent.get('name', 'Agent')}", key=f"chat_{agent.get('id')}"):
                    st.session_state.selected_agent_id = agent['id']
                    st.session_state.current_page = 'chat'
                    st.rerun()
            st.markdown("---")

# --- Sidebar Navigation ---
with st.sidebar:
    st.header("Agent Console")
    if st.button("➕ Create New Agent", use_container_width=True):
        st.session_state.current_page = 'create_agent'
        st.session_state.selected_agent_id = None # Clear selected agent
        st.rerun()

    st.markdown("---")
    st.subheader("Your Agents")

    # Fetch and display agents in sidebar
    agents_in_sidebar = get_agents()
    if not agents_in_sidebar:
        st.info("No agents yet.")
    else:
        for agent in agents_in_sidebar:
            if st.button(agent.get('name', 'Unnamed Agent'), key=f"sidebar_agent_{agent.get('id')}", use_container_width=True):
                st.session_state.selected_agent_id = agent['id']
                st.session_state.current_page = 'chat'
                st.rerun()

# --- Main Content Rendering ---
if st.session_state.current_page == 'create_agent':
    create_agent_page()
elif st.session_state.current_page == 'chat' and st.session_state.selected_agent_id:
    chat_page(st.session_state.selected_agent_id)
else: # Default or if no agent selected for chat
    list_agents_page()

