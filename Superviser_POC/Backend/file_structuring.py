import os
import json
import sqlite3
from pydantic import BaseModel, ValidationError, field_validator,Field
from andromeda import SystemMessage, HumanMessage
from andromeda.core.workflow import WorkflowBuilder
from config import PATH_PROCESSED, PATH_STRUCTURED,COMBINED_INPUT, UNIFIED_CLAIMS_OUT, DB_PATH, get_genai_llm, logger
from models import LossRun
from prompts import EXTRACTION_PROMPT

llm =  get_genai_llm()

MAX_RETRIES = 3

def extract_structured(raw_text: str, source_file: str) -> tuple[LossRun | None, str | None]:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        user_message = f"Source file: {source_file}\n\nDocument text:\n{raw_text}"
        if last_error:
            user_message += (f"Your previous answer failed validation with error:{last_error}."
                "Please fix it and respond again with ONLY the corrected JSON.")
        try:
            response = llm.invoke([
                SystemMessage(content=EXTRACTION_PROMPT),
                HumanMessage(content=user_message)
            ])

            answer = response.text.strip()

            if answer.startswith("```"):
                lines = answer.split("\n")
                answer = "\n".join(lines[1:-1]) if lines[-1].strip().startswith("```") else "\n".join(lines[1:])

            data = json.loads(answer)
            data["source_file"] = source_file
            result = LossRun(**data) 
            return result, None

        except json.JSONDecodeError as e:
            last_error = f"Response was not valid JSON: {e}"
        except ValidationError as e:
            last_error = f"Schema validation failed: {e}"
        except Exception as e:
            last_error = f"LLM call failed: {e}"
        print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {last_error}")
    return None, last_error


def structure_files():
    md_files = [name for name in os.listdir(PATH_PROCESSED) if name.endswith(".md")]
    all_results = []
    failures = []

    for md_file in md_files:
        md_path = os.path.join(PATH_PROCESSED, md_file)

        with open(md_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        result, error = extract_structured(raw_text, source_file=md_file)

        if result:
            all_results.append(result.model_dump())
            out_name = md_file.replace(".md", ".json")
            out_path = os.path.join(PATH_STRUCTURED, out_name)
            with open(out_path, "w") as f:
                json.dump(result.model_dump(), f, indent=2)
            print(f"DOne with {out_name}\n")
        else:
            failures.append({"file": md_file, "error": error})
            print(f"ERROR: {error}\n")

    combined_path = os.path.join(PATH_STRUCTURED, "structured_all_files.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Done files structured successfully.")
    if failures:
        for fail in failures:
            print(f"Failed:- {fail['file']}: {fail['error']}")

############ DB Built ###############################33333333

def step_build_database() -> dict:
    with open(COMBINED_INPUT, "r") as f:
        loss_runs = json.load(f)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE claims (
            claim_number TEXT,
            loss_date TEXT,
            status TEXT,
            cause_of_loss TEXT,
            state TEXT,
            paid_loss REAL,
            paid_expense REAL,
            reserve REAL,
            total_incurred REAL,
            subrogation REAL,
            cat_indicator INTEGER,
            source_file TEXT,
            carrier_name TEXT,
            policy_year TEXT,
            policy_start TEXT,
            policy_end TEXT,
            annual_premium REAL,
            per_occurrence_limit REAL
        )
    """)

    claim_count = 0
    for loss_run in loss_runs:
        policy_year = f"{loss_run.get('policy_start', '?')} - {loss_run.get('policy_end', '?')}"

        for claim in loss_run.get("claims", []):
            cur.execute("""
                INSERT INTO claims (
                    claim_number, loss_date, status, cause_of_loss, state,
                    paid_loss, paid_expense, reserve, total_incurred, subrogation,
                    cat_indicator, source_file, carrier_name, policy_year,
                    policy_start, policy_end, annual_premium, per_occurrence_limit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                claim.get("claim_number"), claim.get("loss_date"), claim.get("status"),
                claim.get("cause_of_loss"), claim.get("state"),
                claim.get("paid_loss", 0), claim.get("paid_expense", 0),
                claim.get("reserve", 0), claim.get("total_incurred", 0),
                claim.get("subrogation", 0), int(bool(claim.get("cat_indicator"))),
                loss_run.get("source_file"), loss_run.get("carrier_name"), policy_year,
                loss_run.get("policy_start"), loss_run.get("policy_end"),
                loss_run.get("annual_premium"), loss_run.get("per_occurrence_limit"),
            ))
            claim_count += 1

    conn.commit()
    conn.close()

    logger.info(f"[build_database]: {DB_PATH} with {claim_count} claim(s)")
    print(f"Database built: {DB_PATH}")
    print(f"  {claim_count} claim(s) loaded into the 'claims' table")

    return {"db_path": DB_PATH, "claim_count": claim_count}

def complete_structure():
    structure_files()
    step_build_database()

if __name__=="__main__":
    complete_structure()
