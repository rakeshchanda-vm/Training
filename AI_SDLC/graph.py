from langgraph.graph import START, END, StateGraph
from config import Workflow, get_llm
import uuid
from prompt import CLASSITY_PROMPT
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import interrupt

# async def resume_or_create_sessions(Workflow):
#     thread_id = str(uuid.uuid4())[:8]
#     requirement = input("Enter you requirements: ")
#     return {'thread_id':thread_id, "user_query":requirement}

async def clarrify_requirement(state:Workflow):
    llm = get_llm()
    prompt = CLASSITY_PROMPT.format(requirement = state.user_query)
    response = llm.invoke([HumanMessage(content= prompt)])
    question = response.content
    print(question)

    answer_raw = interrupt(f"Pls answer the above questions (Press Enter after each type, 'done' to finish)"
                           f"{question}")

    return {
        'clarification_answers': {'raw':answer_raw},
        'message':state.message + [AIMessage(content=question),
                                              HumanMessage(content=str(answer_raw))],
        'status':'clarrified'
    }

def create_graph(checkpointer):
    builder = StateGraph(Workflow)
    builder.add_node("clarrify",clarrify_requirement)

    builder.add_edge(START,"clarrify")
    builder.add_edge("clarrify",END)

    return builder.compile(checkpointer= checkpointer)
    
