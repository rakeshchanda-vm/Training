import requests
import wikipedia
import arxiv
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor


llm = ChatOllama(model="llama3.2:3b", temperature=0)


@tool
def search_wikipedia(topic:str)->str:
    """
    Search Wikipedia for background information.

    Use when you need:
    - definitions
    - history
    - general knowledge
    """
    try:
        result = wikipedia.summary(topic,sentences=5)
        return result
    
    except Exception as e:
        return (f"Wikipedia search failed: {e}")


@tool
def search_research_papers(topic:str)->str:
    """
    Search academic research papers.

    Use for:
    - scientific research
    - AI papers
    - technical topics
    """
    try:
        search = arxiv.Search( query=topic, max_results=3 )
        papers=[]

        for paper in search.results():
            papers.append(
                f"""
                Title: {paper.title}

                Summary: {paper.summary[:500]}

                URL: {paper.entry_id}
                """
            )
        return "\n\n".join(papers)
    
    except Exception as e:
        return (f"Research paper search failed {e}")


@tool
def web_search(query:str)->str:
    """
    Search internet for latest information.

    Use when:
    - recent information needed
    - current events
    - latest technology updates

    """
    try:

        url = ("https://duckduckgo.com/html/")

        response=requests.post(url,
                            data={"q":query},timeout=10)
        text=response.text
        return text[:3000]

    except Exception as e:
        return (f"Web search failed {e}")


tools=[
    search_wikipedia,
    search_research_papers,
    web_search
    ]


prompt = ChatPromptTemplate.from_messages([
(
"system","""
            You are an expert AI Research Agent.
            Your job is to perform deep research.
            For a given topic do check for all wikipedia, web and science papers exposed as tools.
            DO NOT USE YOUR OWN PRE-EXISTING KNOWLEDGE.
            RESPOND ONLY BASED ON CONTEXT COLLECTED USING TOOLS SEARCH.
            The response should be sound like a great writer showing his reseach in a great format.
            ALWAYS PROVIDE CITATION / SOURCES TO CREATE TRUST IN ANSWER.
            ALWAYS INVOKE ALL TOOLS FOR BETTER CONEXT GATHERING

            Research methodology:

            1. Understand the research question.
            2. Break the question into smaller topics.
            3. Select appropriate tools.
            4. Collect information from multiple sources.
            5. Compare information.
            6. Produce a structured research report.

            Rules:

            - Do not answer from memory.
            - Always use tools for factual research.
            - Prefer multiple sources.
            - Mention sources used.
            - Highlight uncertainty.
            - Separate facts from opinions.
            - If refering research paper, provide the url or source of it.

            **********************
            IMPORTANT
            Final report format:

            # Executive Summary
            # Key Findings
            # Detailed Analysis
            # Technical Explanation
            # Sources
            *********************

            """
),

("human","{input}"),

("placeholder","{agent_scratchpad}")]

)

agent=create_tool_calling_agent(
    llm,
    tools,
    prompt
    )

research_agent = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10
    )


question=input("Topic of Research: ")

result = research_agent.invoke({"input":question})

print(result["output"])