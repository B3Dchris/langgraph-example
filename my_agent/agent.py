from typing import TypedDict, Literal, List, Tuple
from langgraph.graph import StateGraph, END
from langchain.llms import OpenAI

##############################################################################
# 1) Agent State & Config
##############################################################################

class AgentState:
    """
    Holds information about the current conversation:
    - user_input: the latest user message
    - last_llm_output: the LLM's most recent response
    - last_tool_output: output from any tool node
    - conversation_history: a list of (role, text) pairs for memory
    """
    def __init__(self, user_input: str = ""):
        self.user_input: str = user_input
        self.last_llm_output: str = ""
        self.last_tool_output: str = ""
        self.conversation_history: List[Tuple[str, str]] = []

class GraphConfig(TypedDict):
    # Example config if you want to switch between Anthropic / OpenAI, etc.
    model_name: Literal["anthropic", "openai"]

##############################################################################
# 2) Node Functions
##############################################################################

def call_model(state: AgentState) -> AgentState:
    """
    Calls GPT-4 via LangChain, storing the result in state.last_llm_output.
    """
    import os
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_api_key:
        state.last_llm_output = "Error: OPENAI_API_KEY not set."
        return state

    llm = OpenAI(
        openai_api_key=openai_api_key,
        temperature=0.7,
        model_name="gpt-4"  # Adjust if you prefer "gpt-3.5-turbo"
    )

    # For simplicity, feed only the latest user_input.
    # You could also include state.conversation_history in the prompt.
    prompt = f"User: {state.user_input}\nAssistant:"
    response = llm(prompt)
    state.last_llm_output = response
    return state

def update_memory(state: AgentState) -> AgentState:
    """
    Appends the latest user message + LLM response to conversation_history.
    """
    user_msg = state.user_input
    llm_reply = state.last_llm_output
    state.conversation_history.append(("user", user_msg))
    state.conversation_history.append(("assistant", llm_reply))
    return state

def calc_node(state: AgentState) -> AgentState:
    """
    Naive calculator node. Uses Python eval() to handle user_input as an expression.
    WARNING: eval() can be unsafe in production if inputs are untrusted.
    """
    user_msg = state.user_input.strip()
    try:
        result = eval(user_msg)  # Not safe for production!
        state.last_tool_output = f"Calculator result: {result}"
    except Exception as e:
        state.last_tool_output = f"Calculator error: {e}"
    return state

def tool_node(state: AgentState) -> AgentState:
    """
    Stub for a generic tool. In real code, you'd call an external API or perform
    some action. We'll just pretend here.
    """
    state.last_tool_output = "Pretend we used a tool here!"
    return state

def should_continue_or_calc(state: AgentState) -> str:
    """
    Decide next step:
    - If user input looks like a math expression or contains 'calc', go to 'calc'.
    - If user says 'quit' or 'end', return 'end'.
    - Otherwise, continue normal flow.
    """
    user_msg = state.user_input.lower()
    if "calc" in user_msg or "calculate" in user_msg:
        return "calc"
    if "quit" in user_msg or "end" in user_msg:
        return "end"
    return "continue"

##############################################################################
# 3) Build the Workflow (LangGraph)
##############################################################################

# Create a new StateGraph for our AgentState, optionally using GraphConfig
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("update_memory", update_memory)
workflow.add_node("calc", calc_node)
workflow.add_node("tool", tool_node)

# Entry point is "agent"
workflow.set_entry_point("agent")

# After "agent", decide: 'calc', 'continue', or 'end'
workflow.add_conditional_edges(
    "agent",
    should_continue_or_calc,
    {
        "calc": "calc",
        "continue": "update_memory",
        "end": END
    }
)

# After updating memory, go to the generic "tool" node
workflow.add_edge("update_memory", "tool")

# After the tool, loop back to "agent"
workflow.add_edge("tool", "agent")

# After the calculator, also loop back to "agent"
workflow.add_edge("calc", "agent")

# Compile into a LangChain Runnable
graph = workflow.compile()

##############################################################################
# 4) Usage
##############################################################################
# You can now import 'graph' elsewhere or deploy it on LangSmith with a
# `langgraph.json` that references this module's workflow.
#
# Example local usage:
#   state = AgentState("calc 2+2")
#   new_state = graph.invoke(state)
#   print(new_state.last_tool_output)  # Should show "Calculator result: 4"