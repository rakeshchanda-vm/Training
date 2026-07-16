import asyncio
from config import get_checkpointer, list_sessions, Workflow
import uuid
from graph import create_graph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import os
from langgraph.types import Command

DB = os.path.join(os.path.dirname(__file__),"checkpointers.db")



BANNER = """
============================================================================
====================== RAKESH AI CODING ASSISTANT ==========================
============================================================================
"""

def resume_or_create_sessions():
    thread_id = str(uuid.uuid4())[:8]
    requirement = input("Enter you requirements: ")
    return {'thread_id':thread_id, "user_query":requirement}


async def run():
    print(BANNER)
    # checkpointer = get_checkpointer()
    async with AsyncSqliteSaver.from_conn_string(DB) as checkpointer:
        graph = create_graph(checkpointer)

        session = resume_or_create_sessions()

        thread_id = session["thread_id"]
        user_query = session["user_query"]

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        message = await graph.ainvoke(
            {'thread_id':thread_id,
             "user_query": user_query},
            config=config
        )

        while "__interrupt__" in message:

            question = message["__interrupt__"][0].value
            print("\nAI Question:")
            print(question)

            answer = input("\nYour answer: ")

            message = await graph.ainvoke(
                Command(resume=answer),
                config=config
            )

        print("\nFinal message:")
        print(message)

        print(Workflow)


if __name__ == '__main__':
    asyncio.run(run())