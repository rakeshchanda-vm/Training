from langchain_ollama import ChatOllama
from pprint import pprint

llm = ChatOllama(model="llama3.2:3b",temperature=0.2)

# result = llm.invoke("Tell me about India in 2-3 lines")

# print(result.content)
# print("******************")
# print(result.response_metadata)
# print("******************")

from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

# response = llm.invoke([
#     SystemMessage(content="You are a Python teacher."),
#     HumanMessage(content="Explain decorators.")
# ])

# print(response)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful tutor."),
    ("human", """Explain {topic} in 1-2 lines.
     
     Response  look like 
     {topic}: expalantion
     """).partial(
    format_instructions=parser.get_format_instructions())
])
chain = prompt | llm | parser
messages = chain.invoke({
    "topic": "Neural Networks"
})
# print(messages.content)
print(messages.reply)

