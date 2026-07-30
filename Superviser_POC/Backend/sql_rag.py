import asyncio
import sqlite3
from langchain_core.tools import tool
from andromeda import HumanMessage
from config import DB_PATH, get_agent_tools, logger
from prompts import SQL_PROMPT
import file_structuring as file_structuring
from tools import get_schema, run_query

def build_sql_rag_agent():
    return get_agent_tools(name="sql_rag_AGENT", prompt=SQL_PROMPT, tools=[get_schema, run_query])

async def aask(question: str, agent=None) -> str:
    agent = build_sql_rag_agent()
    reply = await agent.ainvoke([HumanMessage(content=question)])
    return reply[-1].content

def ask(question: str, agent=None) -> str:
    return asyncio.run(aask(question, agent=agent))

async def asql_agent():
    # file_structuring.build_database()
    agent = build_sql_rag_agent()

    print("\nReady. Ask a question (or 'quit').\n")
    while True:
        question = input("Underwriter--> ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if question:
            logger.info(f"[sql_rag] Q: {question}")
            answer = await aask(question, agent=agent)
            logger.info(f"[sql_rag] A: {answer}")
            print(f"\nAgent--> {answer}\n")

async def sql_agent():
    await asql_agent()

if __name__ == "__main__":
    asyncio.run(sql_agent())