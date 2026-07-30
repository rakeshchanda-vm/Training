import json
import asyncio
import os
from andromeda import HumanMessage
from andromeda.core.workflow import WorkflowBuilder
from config import get_agent, logger
from prompts import ANALYTICS_PROMPT
from config import UNIFIED_CLAIMS_OUT, RISK_METRICS_OUT, LLM_REPORT_OUT

def step_llm_report(state: dict) -> dict:
    with open(UNIFIED_CLAIMS_OUT, "r") as f:
        unified = json.load(f)
    with open(RISK_METRICS_OUT, "r") as f:
        risk_metrics = json.load(f)

    payload = {"submission_summary": unified["submission_summary"],
        "risk_metrics": risk_metrics,
        "claims": unified["unified_claims"]}

    user_message = f"""Here is the complete computed data for this submission. Write the full analytics
        report using ONLY these numbers - do not calculate anything new.\n\n
        {json.dumps(payload, indent=2)}
        """

    logger.info("[analytics] generating LLM-driven analytics report...")
    agent = get_agent(name = "ANALYTICS_AGENT", prompt = ANALYTICS_PROMPT)
    reply = agent.invoke([HumanMessage(content=user_message)])
    report_text = reply[-1].content

    with open(LLM_REPORT_OUT, "w") as f:
        f.write(report_text)
    logger.info(f"[analytics] saved to {LLM_REPORT_OUT}")
    return {"report_text": report_text, "report_path": LLM_REPORT_OUT}

def aggregate_llm_report():
    pipeline = WorkflowBuilder(name="AnalyticsReportLLMPipeline")
    (
        pipeline
        .start("llm_report").run(step_llm_report)
    )
    result = pipeline.execute(state={})
    return result["report_text"]

def generate_llm_report():
    return aggregate_llm_report()
    # asyncio.run(aggregate_llm_report())

if __name__ == "__main__":
    aggregate_llm_report()