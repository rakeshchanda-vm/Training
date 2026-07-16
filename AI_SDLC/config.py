from pydantic import BaseModel,Field
from langchain_ollama import ChatOllama
from typing import TypedDict,Optional,Any,List
from langgraph.checkpoint.sqlite import SqliteSaver
import os
import sqlite3

llm = ChatOllama(model = 'qwen2.5-coder:7b', temperature = 0)

def get_llm():
    return llm

def merge_dicts( existing: dict[str, Any],new: dict[str, Any]) -> dict[str, Any]:
    return {**existing,**new}

def keep_latest(old: Optional[str],new: Optional[str]) -> Optional[str]:
    return (new if new is not None else old)

# class Workflow(BaseModel):
#     thread_id: str
#     user_query : str
#     message : list[str]
#     clarification_answers : dict
#     code_plan : str
#     files_to_generate : list[str]
#     generated_files : dict
#     review : str
#     iteration : int
#     output_foler : str
#     entry_point : str
#     execution_result : dict
#     error : dict
#     fix_iteration : int
#     execution_history : list
#     thread_id : str
#     messages : list
#     status : str


class Workflow(BaseModel):
    thread_id: str
    user_query: str
    message: list[str] = Field(default_factory=list)
    clarification_answers: dict[str, Any] = Field(default_factory=dict)
    code_plan: str = ""
    files_to_generate: list[str] = Field(default_factory=list)
    generated_files: dict[str, Any] = Field(default_factory=dict)
    review: str = ""
    iteration: int = 0
    output_folder: str = ""
    entry_point: str = ""
    execution_result: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] = Field(default_factory=dict)
    fix_iteration: int = 0
    execution_history: list[Any] = Field(default_factory=list)
    status: str = "START"

DB = os.path.join(os.path.dirname(__file__),"checkpointers.db")

def get_checkpointer()->SqliteSaver:
    return SqliteSaver.from_conn_string(DB)

def list_sessions()-> list[dict]:
    if os.path.exists(DB):
        return []
    
    try:
        conn = sqlite3.connect(DB)
        cursor = conn.cursor()

        cursor.execute("SELECT thread_id, MAX(checkpoint_id) as last_checkpoint" \
        "FROM checkpoints GROUP BY thread_id order by last_checkpoint DESC"
                       )
        
        rows = cursor.fetchall()

        conn.close()
        return [{"thread_id":r[0], "last_checkpoint":r[1]} for r in rows]
    except:
        return []
