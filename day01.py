### Day 01 of Coding ###
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field


text_laoding = TextLoader("./the-verdict.txt")
docs = text_laoding.load()
docu_text = docs[0].page_content

class DocumentParser(BaseModel):
    topic:str = Field(description="Main topic or title of document")
    desc:str = Field(description="Short explanation in 4-5 lines")


llm = OllamaLLM(model="qwen3:4b",temperature=0.4,max_tokens= 500)


parser = StrOutputParser()
new_parser = PydanticOutputParser(pydantic_object=DocumentParser)

format_instructions = new_parser.get_format_instructions()


system_prompt = ChatPromptTemplate.from_template("""
You are a expert summarizer. Help us with summarizing following document in 2-3 lines.                                                
                                                 {format_instructions}
                                                 
                                                 Document: {document}
                                                 """)

chain = system_prompt | llm | new_parser

message = chain.invoke({
    "document": docu_text,
    "format_instructions":format_instructions
})

print(message)
