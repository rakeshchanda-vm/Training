import sqlite3
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import sql_rag

@pytest.fixture
def test_db(tmp_path, monkeypatch):
    db = tmp_path / "test.db"

    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE claims (claim_number TEXT, total_incurred REAL, status TEXT)"
    )
    conn.executemany(
        "INSERT INTO claims VALUES (?, ?, ?)",
        [
            ("CL-2022-00004", 90359.71, "Open"),
            ("CL-2020-00005", 7841.32, "Closed"),
        ],
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(sql_rag, "DB_PATH", str(db))
    return db

def test_get_schema(test_db):
    cols = sql_rag.get_schema.invoke({})
    assert {c["column"] for c in cols} == {
        "claim_number",
        "total_incurred",
        "status",
    }

def test_run_query_select(test_db):
    result = sql_rag.run_query.invoke(
        {"sql": "SELECT claim_number FROM claims WHERE status='Open'"}
    )

    assert result["rows"] == [{"claim_number": "CL-2022-00004"}]

@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE claims",
        "INSERT INTO claims VALUES ('x',1,'Open')",
        "DELETE FROM claims",
        "SELECT * FROM claims; DROP TABLE claims",
    ],
)
def test_run_query_blocks_non_select(sql, test_db):
    result = sql_rag.run_query.invoke({"sql": sql})
    assert "error" in result
    conn = sqlite3.connect(test_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 2
    finally:
        conn.close()

class StubAgent:
    def __init__(self, answer):
        self.answer = answer
        self.messages = None
    async def ainvoke(self, messages):
        self.messages = messages
        return [type("Reply", (), {"content": self.answer})()]


