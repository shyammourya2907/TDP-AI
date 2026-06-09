import operator
from typing import Annotated, Any, Dict, List, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langgraph.graph import StateGraph, END
from agents import get_research_agent, get_math_agent

# 1. Define Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# 2. Node Functions
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # Convert agent output to a message
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

def create_supervisor(llm: ChatOpenAI, members: list[str]):
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the following workers: {members}. "
        "Given the following user request, respond with the worker to act next. "
        "Each worker will perform a task and respond with their results and status. "
        "When finished, respond with FINISH."
    )
    
    options = ["FINISH"] + members
    
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                }
            },
            "required": ["next"],
        },
    }
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"
        )
    ]).partial(options=str(options), members=", ".join(members))
    
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

# 3. Build Graph
def build_multi_agent_graph():
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    research_agent = get_research_agent(llm)
    math_agent = get_math_agent(llm)
    
    supervisor_chain = create_supervisor(llm, ["Research", "Math"])
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("Research", lambda state: agent_node(state, research_agent, "Research"))
    workflow.add_node("Math", lambda state: agent_node(state, math_agent, "Math"))
    workflow.add_node("Supervisor", supervisor_chain)
    
    for member in ["Research", "Math"]:
        workflow.add_edge(member, "Supervisor")
        
    conditional_map = {k: k for k in ["Research", "Math"]}
    conditional_map["FINISH"] = END
    
    workflow.add_conditional_edges("Supervisor", lambda x: x["next"], conditional_map)
    workflow.set_entry_point("Supervisor")
    
    return workflow.compile()
