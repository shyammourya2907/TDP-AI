from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from supervisor import build_multi_agent_graph

load_dotenv()

def main():
    print("Initializing Multi-Agent Supervisor System...")
    graph = build_multi_agent_graph()
    
    query = "Find the current population of Tokyo and multiply it by 10."
    print(f"\nUser Query: {query}\n")
    print("-" * 50)
    
    # Run the graph
    for s in graph.stream({"messages": [HumanMessage(content=query)]}):
        if "__end__" not in s:
            for k, v in s.items():
                print(f"[{k}] >>\n")
                # Ensure we handle the messages list safely
                if "messages" in v and isinstance(v["messages"], list):
                     for m in v["messages"]:
                         print(m.content)
                else:
                     print(v)
                print("-" * 50)

if __name__ == "__main__":
    main()
