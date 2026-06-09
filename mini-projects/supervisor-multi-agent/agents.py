from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

# Basic Search Tool
search_tool = TavilySearchResults(max_results=2)

# Custom Math Tool
@tool
def calculate_math(expression: str) -> str:
    """Evaluates a mathematical expression safely."""
    try:
        # NOTE: eval is used here for demonstration, do not use raw eval in production
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    """Helper function to create an agent executor."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

def get_research_agent(llm: ChatOpenAI):
    """Creates a research agent with search capabilities."""
    system_prompt = (
        "You are a Research Assistant. Your job is to search the web for accurate and "
        "up-to-date information. Summarize your findings clearly."
    )
    return create_agent(llm, [search_tool], system_prompt)

def get_math_agent(llm: ChatOpenAI):
    """Creates a math agent."""
    system_prompt = (
        "You are a Math Assistant. Your job is to perform mathematical calculations "
        "and data analysis accurately. Use the calculate_math tool when needed."
    )
    return create_agent(llm, [calculate_math], system_prompt)
