import sys
import tempfile
import subprocess
import re
from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START

# llm = ChatOllama(model="llama3.2:3b",temperature=0)
llm = ChatOllama(model="qwen2.5-coder:7b",temperature=0)



class ReviewState(TypedDict):
    file_path: str
    code: str
    bug_review: str
    security_review: str
    quality_review: str
    execution_result: str
    error_details: str
    runtime_review: str
    final_report: str


def load_code(state):
    with open(state["file_path"], "r") as file:
        code = file.read()
    return {"code": code}


def bug_agent(state):
    prompt = f"""
        You are a senior Python debugging expert.
        Analyze this Python code.
        Find:
        - logical mistakes
        - runtime risks
        - missing validations
        - incorrect assumptions

        For every issue provide:
        Problem:
        Severity:
        Explanation:
        Fix:

        CODE:
            {state["code"]}

        """
    response = llm.invoke(prompt)
    return {"bug_review": response.content}


def security_agent(state):
    prompt = f"""
        You are` a Python security engineer.
        Review this code for security problems.
        Look for:
        - hardcoded passwords
        - API keys
        - unsafe functions
        - injection risks
        - insecure practices

        Provide:
        Issue:
        Severity:
        Explanation:
        Recommendation:

        CODE: 
        `{state["code"]}`
        """
    response = llm.invoke(prompt)
    return {"security_review": response.content}


def quality_agent(state):
    prompt = f"""
        You are a senior Python architect.
        Review this code quality.
        Check:
        - readability
        - maintainability
        - complexity
        - Python best practices
        - design problems

        Provide:
        Problem:
        Impact:
        Recommendation:

        CODE:
        {state["code"]}
        """
    response = llm.invoke(prompt)
    return {"quality_review": response.content}


def execute_code(state):
    code = state["code"]

    with tempfile.NamedTemporaryFile(suffix=".py",mode="w",delete=False) as file:
        file.write(code)
        temp_file = file.name
    try:
        result = subprocess.run(
            [
                "python",
                temp_file
            ],
            capture_output=True,
            text=True,
            timeout=300
        )

        execution_output = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }

    except subprocess.TimeoutExpired:
        execution_output = {"stderr":"Program execution exceeded 5 seconds timeout"}

    return {"execution_result":str(execution_output)}


def extract_error(state):
    execution = state["execution_result"]
    error_info = {
        "error_type": "None",
        "line_number": None,
        "code_line": None,
        "message": execution
    }

    if "Traceback" in execution:
        match = re.search(
            r'line (\d+)',
            execution
        )

        if match:
            line_number = int(
                match.group(1)
            )
            lines = state["code"].split("\n")
            if line_number <= len(lines):
                error_info["line_number"] = line_number
                error_info["code_line"] = (lines[line_number-1])

        error_info["error_type"] = (
            execution.split(":")[-2]
            if ":" in execution
            else "Runtime Error"
        )

    return {"error_details":str(error_info)}


def runtime_agent(state):
    prompt = f"""
        You are a Python runtime debugging expert.
        Analyze this execution problem.
        SOURCE CODE:
        {state["code"]}

        ERROR DETAILS:
        {state["error_details"]}

        Explain:
        1. What failed?
        2. Exact problematic code line.
        3. Why it hgraphened.
        4. Recommended fix.

        Highlight the problematic code.
        """
    response = llm.invoke(prompt)
    return {"runtime_review":response.content}


def final_agent(state):
    prompt = f"""
        You are the lead software engineer.
        Create the final code review report.
        Include:

        ## Summary
        ## Runtime Errors
        {state["runtime_review"]}

        ## Bug Review
        {state["bug_review"]}

        ## Security Review
        {state["security_review"]}

        ## Code Quality Review
        {state["quality_review"]}

        ## Recommended Fixes
        Make important runtime issues clearly visible.
        """
    response = llm.invoke(prompt)

    return {"final_report":response.content}


builder = StateGraph(
    ReviewState
)


builder.add_node("loader",load_code)
builder.add_node("bug_checker",bug_agent)
builder.add_node("security_checker",security_agent)
builder.add_node("quality_checker",quality_agent)
builder.add_node("executor",execute_code)
builder.add_node("error_parser",extract_error)
builder.add_node("runtime_checker",runtime_agent)
builder.add_node("final_report",final_agent)

builder.add_edge(START,"loader")
builder.add_edge("loader","bug_checker")
builder.add_edge("bug_checker","security_checker")
builder.add_edge("security_checker","quality_checker")
builder.add_edge("quality_checker","executor")
builder.add_edge( "executor","error_parser")
builder.add_edge("error_parser","runtime_checker")
builder.add_edge("runtime_checker","final_report")
builder.add_edge("final_report", END)

graph = builder.compile()

result = graph.invoke({"file_path":sys.argv[1]})

print("\n")
print("="*70)
print(" AI CODE REVIEW REPORT ")
print("="*70)

print(result["final_report"])