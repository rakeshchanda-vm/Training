from andromeda.utils import get_chat_model
from andromeda.config import ModelConfig, AgentConfig
from andromeda.core.agent import Agent
import os
import logging

########################## PATH DEFINATION ########################
PATH_DATA_FOLDER = "/home/rchanda/TestFolder/POCs_Folder/Training_POC_Insurance/Data"
PATH_STRUCTURED = f"{PATH_DATA_FOLDER}/Structured"
PATH_OUTPUT = f"{PATH_DATA_FOLDER}/Output"
PATH_PROCESSED = f"{PATH_DATA_FOLDER}/Processed"
PATH_RAW = f"{PATH_DATA_FOLDER}/RAW"

COMBINED_INPUT = os.path.join(PATH_STRUCTURED, "structured_all_files.json")
UNIFIED_CLAIMS_OUT = os.path.join(PATH_OUTPUT, "unified_claims.json")
RISK_METRICS_OUT = os.path.join(PATH_OUTPUT, "risk_metrics.json")
MEMO_OUT = os.path.join(PATH_OUTPUT, "underwriting_memo.md")
LLM_REPORT_OUT = os.path.join(PATH_OUTPUT, "analytics_report_llm.md")
DB_PATH = os.path.join(PATH_OUTPUT, "submission.db")

LARGE_LOSS_THRESHOLD_PCT = 0.50

##################### LOGGING ####################################3
LOG_PATH = os.path.join(PATH_OUTPUT, "pipeline.log")

def _configure_logging() -> logging.Logger:
    log = logging.getLogger("UNDERWRITER_AGENT")
    if log.handlers:  # config.py can be imported more than once per process
        return log
    log.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    log.addHandler(file_handler)
    return log

logger = _configure_logging()

##################### LLM and AGENT ##############################
OCR_MODEL = "minicpm-v:latest"
GEN_MODEL = "gpt-oss:20b"
# GEN_MODEL = "llama3.2:3b"
PROVIDER = "litellm"
# PROVIDER = "ollama"

def get_genai_llm():
    return get_chat_model(model_config=ModelConfig(name=GEN_MODEL,provider=PROVIDER, temperature=0))

def get_agent(name:str, prompt:str):
    llm_model = get_genai_llm()
    return Agent(
                AgentConfig(
                    name = name,
                    model = ModelConfig(name=GEN_MODEL,provider=PROVIDER, temperature=0),
                    prompt = prompt,
                )
            )

def get_agent_tools(name:str, prompt:str, tools:list[str]):
    llm_model = get_genai_llm()
    return Agent(
                AgentConfig(
                    name = name,
                    model = ModelConfig(name=GEN_MODEL,provider=PROVIDER, temperature=0),
                    tools = tools,
                    prompt = prompt,
                )
            )