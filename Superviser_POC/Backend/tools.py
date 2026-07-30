from andromeda.tools.toolkit import register_tools
from andromeda.tools import tool
import memo_builder
import analytics
from config import logger
import file_processing
import file_scoring
import file_structuring
import memo_builder
import analytics
from andromeda.core.workflow import WorkflowBuilder
import sqlite3
from config import DB_PATH


BLOCKED_WORDS = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH"]

@tool
def get_schema() -> list[dict]:
    """
    Get the claims table's columns before writing a query.
    """
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(claims)")
    columns = [{"column": row[1], "type": row[2]} for row in cur.fetchall()]
    logger.info(f"[tool:get_schema] Schema: {columns}")
    conn.close()
    return columns

@tool
def run_query(sql: str) -> dict:
    """
    Run a read-only SELECT query against the claims table and get the results.
    """
    if not sql.strip().upper().startswith("SELECT") or any(w in sql.upper() for w in BLOCKED_WORDS):
        return {"error": "Only SELECT queries are allowed."}
    logger.info(f"[tool:run_query] Run Query: {sql}")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cur = conn.cursor()
    try:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return {"rows": [dict(zip(cols, row)) for row in cur.fetchall()]}
    except sqlite3.Error as e:
        return {"error": str(e)}
    finally:
        conn.close()

def step_processing(state: dict) -> dict:
    file_processing.process_files()

def step_structuring(state: dict) -> dict:
    file_structuring.structure_files()

def step_scoring(state: dict) -> dict:
    file_scoring.main()

def build_run_pipeline() -> WorkflowBuilder:
    pipeline = WorkflowBuilder(name="Underwriter_Agent")
    (
        pipeline
        .start("processing").run(step_processing)
        .then("structuring").run(step_structuring)
        .finish("scoring").run(step_scoring)
    )
    return pipeline

@tool
async def run_process_file():
    """
    This tool is for file processing. Whenever new file arrives, this tools helps to process file.
    """
    logger.info(f"[tool:process_file] Processing FILE Started")
    pipeline = build_run_pipeline()
    await pipeline.aexecute(state={})
    logger.info(f"[tool:process_file] Processing FILE Completed")

@tool
def step_memo(state: dict) -> dict:
    """
    This tool helps to create a memo for the claims. Its a brief level analysis of data (short quick analysis)
    """
    logger.info(f"[tool:memo_builder] Memo file building Started")
    memo_builder.generate_memo()
    logger.info(f"[tool:memo_builder] Memo file building Completed")

@tool
def step_report(state: dict) -> dict:
    """
    This tool helps to create a complete analysis for the claims. Its a deep complete analysis of data.
    """
    logger.info(f"[tool:analytical_report] Analytical Report building Started")
    analytics.generate_llm_report()
    logger.info(f"[tool:analytical_report] Analytical Report building Completed")