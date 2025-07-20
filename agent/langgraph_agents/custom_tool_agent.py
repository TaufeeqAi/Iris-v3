import logging
import json
import re
from typing import List, Any, TypedDict, Annotated, Dict, Optional

from langchain_groq import ChatGroq
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END

# ADK imports
from google.adk.agents.base_agent import BaseAgent
from google.adk.runners import Runner as AgentRuntime
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.models.agent_config import Message as AppMessage # Renamed for clarity to avoid conflict
from pydantic import Field # Import Field for explicit Pydantic field declaration


logger = logging.getLogger(__name__)


# ——— Shared utilities ——————————————————————————————————————————————

# Define the AgentState for the custom LangGraph agent
class AgentState(TypedDict):
    """
    Represents the state of the agent's conversation.
    - messages: A list of messages in the conversation history.
    """
    messages: Annotated[List[Any], lambda x, y: x + y] # Accumulate messages


MAX_HISTORY_MESSAGES = 10
MAX_TOOL_OUTPUT_CHARS = 1500 # Roughly 300-500 words, depending on character density
# Regex to find and remove the specific <tool-use> tags
TOOL_USE_TAG_REGEX = re.compile(r'<tool-use>.*?<\/tool-use>\s*')

def _truncate_tool_output(output: Any, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """
    Truncates a tool's output if it's too long, or summarizes it if it's a known structured type.
    This helps prevent context window overflow.
    """
    output_str = str(output)
    
    # Attempt to parse as JSON for more intelligent summarization/truncation
    try:
        json_output = json.loads(output_str)
        
        # Specific handling for news articles for making News summary
        if isinstance(json_output, dict) and "articles" in json_output and isinstance(json_output["articles"], list):
            headlines = [art.get("headline", "No headline") for art in json_output["articles"][:5]] # Take top 5 headlines
            summary_str = f"Found {json_output.get('news_count', len(json_output['articles']))} news articles. Top headlines: {'; '.join(headlines)}"
            if len(json_output["articles"]) > 5:
                summary_str += f" (and {len(json_output['articles']) - 5} more...)"
            return summary_str
        
        # Specific handling for multiple stocks
        if isinstance(json_output, dict) and "data" in json_output and isinstance(json_output["data"], dict):
            stock_summaries = []
            for symbol, data in json_output["data"].items():
                if data.get("status") == "success" and data.get("current_price") is not None:
                    stock_summaries.append(f"{symbol}: {data['current_price']:.2f}")
                else:
                    stock_summaries.append(f"{symbol}: Error or N/A")
            return f"Fetched quotes for {len(json_output['data'])} stocks: {', '.join(stock_summaries)}"

        # Default JSON summarization: just return a snippet or a simplified representation
        if len(output_str) > max_chars:
            return f"Large JSON output (truncated): {output_str[:max_chars//2]}...{output_str[-max_chars//2:]}"
        return output_str

    except json.JSONDecodeError:
        # Not a JSON, just truncate plain string
        if len(output_str) > max_chars:
            return f"{output_str[:max_chars]}... (truncated)"
        return output_str


# ——— ADK-wrapped LangGraph agent —————————————————————————————————————

class LangGraphADKAgent(BaseAgent):
    """
    A custom agent that uses LangGraph for its internal logic and integrates
    with Google ADK's runtime and FastMCP for tooling.
    """
    # Declare these as Pydantic fields.
    # Pydantic will handle their assignment and validation automatically
    # when the instance is created via keyword arguments in __init__.
    agent_id: str = Field(..., description="Unique ID for the agent.")
    agent_name: str = Field(..., description="Name of the agent.")
    agent_bio: str = Field(..., description="Short description of the agent's capabilities.")
    agent_persona: str = Field(..., description="The persona/system prompt for the LLM.")
    llm: ChatGroq = Field(..., description="The initialized ChatGroq LLM instance.")
    mcp_client: MultiServerMCPClient = Field(..., description="The FastMCP client for tool access.")
    adk_runtime: Optional[AgentRuntime] = Field(default=None, description="The ADK AgentRuntime instance.")
    tools: List[BaseTool] = Field(..., description="List of LangChain tools available to the agent.")
    # New: Declare langgraph_runnable as a Pydantic Field
    langgraph_runnable: Any = Field(default=None, description="The compiled LangGraph runnable.")


    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        agent_bio: str,
        agent_persona: str,
        llm: ChatGroq,
        mcp_client: MultiServerMCPClient,
        tools: List[BaseTool],
        adk_runtime: Optional[AgentRuntime] = None, # Make it optional in the signature too
        **kwargs: Any, # Capture any extra kwargs
    ):
        # Construct the dictionary of ALL fields that Pydantic needs to validate and assign.
        # This includes fields inherited from BaseAgent ('name', 'description')
        # and new fields defined on LangGraphADKAgent.
        data_for_pydantic = {
            "name": agent_name,  # Map to BaseAgent's 'name' field
            "description": agent_bio, # Map to BaseAgent's 'description' field
            "agent_id": agent_id,
            "agent_name": agent_name,
            "agent_bio": agent_bio,
            "agent_persona": agent_persona,
            "llm": llm,
            "mcp_client": mcp_client,
            "adk_runtime": adk_runtime, # Pass the potentially None value here
            "tools": tools,
            "langgraph_runnable": None, # Initialize as None, will be set after super().__init__
            **kwargs, # Include any other keyword arguments passed
        }
        
        # Call the parent's __init__ with the combined data.
        # Pydantic's BaseModel.__init__ will handle the validation and assignment
        # for all fields declared on LangGraphADKAgent (including inherited ones).
        super().__init__(**data_for_pydantic)
        
        # Build LangGraph workflow
        # Access the fields via self.field as they are now Pydantic fields.
        # This assignment will now work because langgraph_runnable is a declared Field.
        self.langgraph_runnable = self._compile_langgraph(
            llm=self.llm,
            tools=self.tools,
            system_prompt=self.agent_persona,
            agent_name=self.agent_name
        )
        logger.info(f"[{self.agent_name}] LangGraph-ADKAgent initialized with PeerID: {self.agent_id}")


    def _compile_langgraph(self, llm: ChatGroq, tools: List[BaseTool], system_prompt: str, agent_name: str) -> Any:
        """
            Creates and compiles a custom LangGraph agent for tool calling.

            Args:
                llm: The initialized ChatGroq LLM.
                tools: A list of LangChain BaseTool instances.
                system_prompt: The system prompt for the LLM.
                agent_name: The name of the agent.

            Returns:
                A compiled LangGraph agent runnable.
        """
        logger.info(f"Building custom LangGraph agent '{agent_name}'...")
        # Bind tools to the LLM
        llm_with_tools = llm.bind_tools(tools)

        # Define the 'call_model' node
        async def call_model(state: AgentState) -> Dict[str, List[Any]]:
            """
            Invokes the LLM with the current conversation history.
            Applies a sliding window to manage context length.
            Decides whether to call a tool or generate a final answer.
            Cleans the final AI message content from internal tool-use tags.
            """
            messages = state['messages']
            
            # Apply sliding window to messages to manage context length.
            recent_messages = messages[-(MAX_HISTORY_MESSAGES - 1):] if len(messages) > (MAX_HISTORY_MESSAGES - 1) else messages
            
            # Prepend the system message to the recent messages for the LLM call.
            full_messages = [SystemMessage(content=system_prompt)] + recent_messages
            
            logger.debug(f"[{agent_name}] Calling LLM with {len(full_messages)} messages (truncated to {MAX_HISTORY_MESSAGES} including system prompt). Messages: {full_messages}")
            response = await llm_with_tools.ainvoke(full_messages)
            logger.debug(f"[{agent_name}] LLM Response (raw): {response}")

            # Post-process the AI message content to remove unwanted tags
            if isinstance(response, AIMessage) and response.content:
                cleaned_content = TOOL_USE_TAG_REGEX.sub('', response.content).strip()
                # If the content becomes empty after cleaning, ensure it's not entirely blank
                if not cleaned_content and response.tool_calls:
                    pass 
                response.content = cleaned_content
                logger.debug(f"[{agent_name}] LLM Response (cleaned): {response}")

            return {"messages": [response]}
        
        # Define the 'call_tool' node
        async def call_tool(state: AgentState) -> Dict[str, List[Any]]:
            """
            Executes the tool calls requested by the LLM and returns their outputs.
            Trun cates large tool outputs before adding them to the state.
            """
            last_message = state['messages'][-1]
            tool_outputs = []
            
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                for tool_call_item in last_message.tool_calls:
                    tool_name = None
                    tool_args = None
                    tool_call_id = None

                    if isinstance(tool_call_item, dict):
                        tool_name = tool_call_item.get('name')
                        tool_args = tool_call_item.get('args', {})
                        tool_call_id = tool_call_item.get('id')
                    else:
                        logger.error(f"[{agent_name}] Unexpected type for tool_call_item: {type(tool_call_item)}. Expected dict-like. Skipping tool call.")
                        continue

                    if not tool_name:
                        logger.error(f"[{agent_name}] Tool name not found in tool call: {tool_call_item}. Skipping tool call.")
                        continue

                    logger.info(f"[{agent_name}] Attempting to call tool: '{tool_name}' with args: {tool_args}")
                    try:
                        tool_to_call = next((t for t in tools if t.name == tool_name), None)
                        if tool_to_call:
                            raw_output = await tool_to_call.ainvoke(tool_args)
                            
                            # --- Apply truncation/summarization here ---
                            processed_output = _truncate_tool_output(raw_output)
                            logger.info(f"[{agent_name}] Tool '{tool_name}' output (processed for context): {processed_output}")
                            tool_outputs.append(ToolMessage(content=processed_output, tool_call_id=tool_call_id))
                        else:
                            error_msg = f"Tool '{tool_name}' not found."
                            logger.error(f"[{agent_name}] {error_msg}")
                            tool_outputs.append(ToolMessage(content=error_msg, tool_call_id=tool_call_id))
                    except Exception as e:
                        error_msg = f"Error calling tool '{tool_name}': {e}"
                        logger.error(f"[{agent_name}] {error_msg}", exc_info=True)
                        tool_outputs.append(ToolMessage(content=error_msg, tool_call_id=tool_call_id))
            else:
                logger.warning(f"[{agent_name}] 'call_tool' node reached without valid tool calls in the last message or last message is not AIMessage. This is unexpected for this graph flow. Last message: {last_message}")
                pass

            return {"messages": tool_outputs}
        

        # Define the conditional edge logic
        def should_continue(state: AgentState) -> str:
            """
            Determines the next step in the graph based on the LLM's output.
            If the LLM requested tool calls, continue to 'call_tool'.
            Otherwise, if it generated a final answer, end the graph.
            """
            last_message = state['messages'][-1]
            # If the last message is an AI message with tool calls, then execute tools
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                logger.debug(f"[{agent_name}] LLM requested tool calls: {last_message.tool_calls}. Transitioning to 'call_tool'.")
                return "continue"
            # Otherwise, the LLM has generated a final answer, so end the graph
            logger.debug(f"[{agent_name}] LLM generated final answer: {last_message.content}. Transitioning to 'end'.")
            return "end"

        # Build the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("call_model", call_model)
        workflow.add_node("call_tool", call_tool)

        # Set entry point: always start by calling the model
        workflow.set_entry_point("call_model")

        # Define edges
        # If call_model generates tool calls, go to call_tool. Otherwise, end.
        workflow.add_conditional_edges(
            "call_model",
            should_continue,
            {"continue": "call_tool", "end": END}
        )
        # After a tool is called, always go back to the model to process the tool output
        workflow.add_edge("call_tool", "call_model")

        # compile the graph for agent runner
        return workflow.compile()
    
    async def _run_async_impl(self, message: Any, context: Any) -> Any:
            """
            Implements the core logic for ADK's BaseAgent.
            This method is called by the ADK AgentRuntime if this agent were registered directly with it.
            It converts the incoming ADK message to a LangChain HumanMessage and invokes the LangGraph.
            """
            logger.info(f"[{self.agent_name}] ADK BaseAgent _run_async_impl called with message: {message}")
            
            input_message_content = ""
            if isinstance(message, str):
                input_message_content = message
            elif isinstance(message, dict) and "text" in message:
                input_message_content = message["text"]
            elif isinstance(message, AppMessage): 
                input_message_content = message.text
            else:
                input_message_content = str(message) 

            user_message = HumanMessage(content=input_message_content)
            initial_langgraph_state = {"messages": [user_message]}

            try:
                # Invoke the LangGraph workflow
                # The `invoke` method of the compiled LangGraph app returns the final state.
                langgraph_output = await self.langgraph_runnable.ainvoke(initial_langgraph_state)
                
                final_response_content = "I'm sorry, I couldn't process that."
                if "messages" in langgraph_output and langgraph_output["messages"]:
                    last_message = langgraph_output["messages"][-1]
                    if isinstance(last_message, AIMessage):
                        final_response_content = last_message.content
                    elif isinstance(last_message, ToolMessage):
                        final_response_content = f"Tool executed: {last_message.content}"
                    else:
                        final_response_content = str(last_message)
                
                logger.info(f"[{self.agent_name}] LangGraph processed message. Final response: {final_response_content[:100]}...")
                
                # Return the response in a format compatible with ADK's expectations (simple string or dict)
                # Since our A2A connector expects common.models.Message, we'll return its dict representation.
                return AppMessage(text=final_response_content).dict()

            except Exception as e:
                logger.error(f"[{self.agent_name}] Error during ADK agent _run_async_impl: {e}", exc_info=True)
                # Return an error message in a compatible format
                return AppMessage(text=f"An error occurred: {e}").dict()


# ——— Factory function ——————————————————————————————————————————————
async def create_custom_tool_agent(
    llm: ChatGroq,
    tools: List[BaseTool],
    system_prompt: str, # This will be used as agent_persona
    agent_name: str,
    agent_id: str,
    mcp_client: MultiServerMCPClient, # Added mcp_client
    adk_runtime: Optional[AgentRuntime] = None # Made optional with default None
) -> LangGraphADKAgent:
    """Factory: returns a LangGraphADKAgent ready for ADK use."""
    # The system_prompt is now passed as agent_persona to LangGraphADKAgent
    return LangGraphADKAgent(
        agent_id=agent_id,
        agent_name=agent_name,
        agent_bio=system_prompt, # Using system_prompt as bio/description for BaseAgent
        agent_persona=system_prompt, # Using system_prompt as persona for LLM
        llm=llm,
        mcp_client=mcp_client,
        tools=tools,
        adk_runtime=adk_runtime, # Pass the adk_runtime parameter
        langgraph_runnable=None # Pass as None initially, will be set in __init__
    )
