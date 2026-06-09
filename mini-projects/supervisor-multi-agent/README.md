# Supervisor Multi-Agent System (LangGraph)

This project demonstrates **Production-level Multi-Agent Optimization** using LangGraph. It sets up a **Supervisor Agent** that acts as a router/manager. 

Instead of one monolithic agent trying to do everything, the Supervisor reads the user's prompt and delegates tasks to specialized worker agents:
1. **Research Agent**: Equipped with web search tools (Tavily).
2. **Math Agent**: Equipped with calculation tools.

## 📁 Files
- `agents.py`: Defines the tools and the setup for the individual worker agents.
- `supervisor.py`: Defines the routing logic and builds the LangGraph `StateGraph`.
- `main.py`: Entry point to execute the graph with a sample query.

## 🚀 Setup & Execution

1. **Install Dependencies**:
   ```bash
   pip install langchain langchain-openai langgraph tavily-python python-dotenv
   ```

2. **Environment Variables**:
   Create a `.env` file in this directory and add your API keys:
   ```env
   OPENAI_API_KEY="your_openai_api_key"
   TAVILY_API_KEY="your_tavily_api_key"
   ```

3. **Run the Multi-Agent System**:
   ```bash
   python main.py
   ```
   *You will see the Supervisor route the request to Research, get the population, route to Math to multiply it, and then finish.*
