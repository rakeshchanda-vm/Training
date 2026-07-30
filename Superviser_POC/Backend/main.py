import asyncio
from agents import Agent_Summarize
from andromeda import HumanMessage
from andromeda.core.supervisor import Supervisor

MENU = """
==========================================
  Underwriting Risk Triage
==========================================
  Welcome to Underwriting Risk Triage Agent
    Ask any questions or commands next.
==========================================
"""

def amain():
    print(MENU)
    while True:
        query = input("Query--> ").strip().lower()

        if query in ("q", "quit", "exit"):
            break
        else:
            state = {
                "messages": [HumanMessage(content=query)],
                "plan": [],
            }
            result = Agent_Summarize.supervise(state)
            print('*'*100)
            print(result["messages"][-1].content)
            print('='*100)

# def main():
#     asyncio.run(amain())

if __name__ == "__main__":
    amain()