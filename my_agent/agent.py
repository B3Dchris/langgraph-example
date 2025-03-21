from typing import TypedDict, Literal, List, Tuple
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, load_tools
from langchain.memory import ConversationBufferMemory
import os

class AgentState(TypedDict):
    user_input: str
    conversation_history: List[Tuple[str, str]]
    last_output: str

class GraphConfig(TypedDict):
    model_name: Literal["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]

memory = ConversationBufferMemory(memory_key="chat_history")

def call_agent(state: AgentState, config: GraphConfig) -> AgentState:
    model = ChatOpenAI(model=config['model_name'], temperature=0.7)
    tools = load_tools(["llm-math"], llm=model)

    agent = initialize_agent(
        tools,
        model,
        memory=memory,
        agent="chat-conversational-react-description",
        verbose=False,
    )

    try:
        response = agent.run(input=state["user_input"])
        memory.save_context({"input": state["user_input"]}, {"output": response})
    except Exception as e:
        response = f"Error: {str(e)}"

    state["conversation_history"].append((state["user_input"], response))
    state["last_output"] = response

    return state

workflow = StateGraph(AgentState, config_schema=GraphConfig)
workflow.add_node("agent", call_agent)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

graph = workflow.compile()