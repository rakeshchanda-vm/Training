import asyncio
import json
import os
from andromeda import HumanMessage
from andromeda.config import AgentConfig, ModelConfig
from andromeda.core.agent import Agent
from andromeda.core.workflow import WorkflowBuilder
from config import PATH_OUTPUT, RISK_METRICS_OUT, UNIFIED_CLAIMS_OUT, MEMO_OUT, get_agent, logger
from prompts import MEMO_PROMPT


def step_memo(state: dict) -> dict:
    unified_claims = state["unified_claims"]
    risk_metrics = state["risk_metrics"]

    payload = {"submission_summary": unified_claims["submission_summary"],
        "risk_metrics": risk_metrics,
        "claims": unified_claims["unified_claims"]}

    user_message = (f"""Here is the computed data for this submission. Write the underwriting memo for Underwriter.
        "using ONLY these numbers - do not calculate anything new.\n
        {json.dumps(payload, indent=2)}""")

    print("Generating memo...")
    agent = get_agent(name="MEMO_BUILDER", prompt=MEMO_PROMPT)
    reply = agent.invoke([HumanMessage(content=user_message)])
    memo_text = reply[-1].content

    return {"memo_text": memo_text}


def build_memo_pipeline() -> WorkflowBuilder:
    pipeline = WorkflowBuilder(name="MemoPipeline")
    pipeline.start("memo").run(step_memo)
    return pipeline


def agenerate_memo():
    with open(UNIFIED_CLAIMS_OUT, "r") as f:
        unified_claims = json.load(f)

    with open(RISK_METRICS_OUT, "r") as f:
        risk_metrics = json.load(f)

    pipeline = build_memo_pipeline()
    result = pipeline.execute(state={"unified_claims": unified_claims, "risk_metrics": risk_metrics})
    memo_text = result["memo_text"]

    with open(MEMO_OUT, "w") as f:
        f.write(memo_text)
    print(f"Memo saved to: {MEMO_OUT}")

def generate_memo():
    agenerate_memo()

if __name__ == "__main__":
    generate_memo()