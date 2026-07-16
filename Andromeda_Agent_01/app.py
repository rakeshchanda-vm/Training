# from andromeda.config import ModelConfig
# from andromeda.utils import get_chat_model

# llm = get_chat_model(model_config=ModelConfig(name='llama3.2:3b',provider='ollama', output_version='v1',
#                                               temperature=0))

# response = llm.invoke("Give me multiplication table of 25 upto 10 iterations")
# print(response.text)

#################################################################################################################

# from andromeda.core.agent import Agent
# from andromeda.config import ModelConfig, AgentConfig
# from andromeda import HumanMessage
# from pydantic import BaseModel

# agent = Agent(
#     AgentConfig(
#         name = "Agent_01",
#         model= ModelConfig(name="llama3.2:3b", provider='ollama', temperature=0.1),
#         prompt='You are a concise technical assistant.'
#     )
# )
# message = [HumanMessage(content="What is Andromeda in one sentence?")]
# response = agent.invoke(messages=message)
# print(response[-1].content)

###################################################################################################################3

# from typing import Dict, Any

# from andromeda.tools import tool
# from andromeda.core.agent import Agent
# from andromeda.config import AgentConfig, ModelConfig
# from andromeda import HumanMessage
# import asyncio


# @tool
# def echo_tool(text: str) -> Dict[str, Any]:
#     """Echo text back. Replace with your real tool logic."""
#     return {"echo": text}

# async def main():
#     agent = Agent(
#         AgentConfig(
#             name="tool_agent",
#             model=ModelConfig(name="llama3.2:3b", provider="ollama"),
#             # tools=[echo_tool],
#             # prompt="Use tools when they improve correctness. No extra explanations.",
#         )
#     )

#     result = await agent.ainvoke([
#         HumanMessage(content="Give me 3 bullet points about workflow automation.")
#     ])
#     print(result[-1].content)

# if __name__ =='__main__':
#     asyncio.run(main())

##########################################################################################

# from andromeda.core.agent import Agent
# from andromeda.config import AgentConfig, ModelConfig

# agent = Agent(
#     AgentConfig(name="worker", model=ModelConfig(name="llama3.2:3b", provider="ollama"))
# )

# report = agent.invoke("Summarize the top 5 risks in deploying a new API.")
# report1 = agent.task("Summarize the top 5 risks in deploying a new API.")
# report2 = agent.research("Summarize the top 5 risks in deploying a new API.")

# print(report)
# print("="*30)
# print(report1)
# print("="*30)
# print(report2)
# print("="*30)

#############################################################################################

# import asyncio

# from andromeda.core.agent import Agent
# from andromeda.config import AgentConfig, ModelConfig
# from andromeda import HumanMessage


# async def main() -> None:
#     agent = Agent(
#         AgentConfig(name="streamer", model=ModelConfig(name="llama3.2:3b", provider="ollama"))
#     )
#     history = [HumanMessage(content="Explain streaming in plain English in 4-5 lines")]

#     async for event in agent.astream(history, stream_mode="events"):
#         if event.get("event") == "on_chat_model_stream":
#             chunk = event.get("data", {}).get("chunk")
#             if chunk and chunk.content:
#                 print(chunk.content, end="", flush=True)


# if __name__ == "__main__":
#     asyncio.run(main())


##############################################################################
from andromeda.core.agent import Agent
from andromeda.core.supervisor import Supervisor
from andromeda.config import AgentConfig, SupervisorConfig, ModelConfig
from andromeda import HumanMessage

# Specialist worker configs
researcher = AgentConfig(
    name="researcher",
    model=ModelConfig(name="llama3.2:3b", provider="ollama"),
    prompt="Focus on evidence gathering and source quality.",
)
writer = AgentConfig(
    name="writer",
    model=ModelConfig(name="llama3.2:3b", provider="ollama"),
    prompt="Produce clear, concise writing from verified notes.",
)

# Supervisor config
supervisor_cfg = SupervisorConfig(
    name="supervisor",
    model=ModelConfig(name="llama3.2:3b", provider="ollama"),
    prompt="Plan and coordinate agents to fully cover the task.",
    enable_planning=True,
)

supervisor = Supervisor(agents=[researcher, writer], config=supervisor_cfg)

state = {
    "messages": [HumanMessage(content="Research EV trends and draft a short brief.")],
    "plan": [],
}
result = supervisor.supervise(state)
print(result["messages"][-1].content)

