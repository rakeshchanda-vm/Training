from __future__ import annotations
import builtins
import contextlib
import io
import subprocess
import os
import tempfile
import json
from typing import Any, Dict, Tuple, Optional


def mamba_eval(
    code: str,
    _locals: dict[str, Any],
) -> Tuple[str, dict[str, Any]]:
    """
    Execute `code` inside the specified conda/mamba environment and return:

        (stdout_from_code_or_error, {new_variable_name: value, …})

    Requirements
    ------------
    * `mamba` (or `conda`) must be on PATH.
    * The environment `env` must already exist and contain Python.
    """
    # Wrap user code so we can capture stdout **and** the variables it creates
    wrapper = f"""
import builtins, contextlib, io, json, sys
_locals = {_locals}
original_keys = set(_locals.keys())

try:
    with contextlib.redirect_stdout(io.StringIO()) as f:
        exec({code!r}, builtins.__dict__, _locals)
    result = f.getvalue() or "<code ran, no output printed to stdout>"
except Exception as e:
    result = f"Error during execution: {{repr(e)}}"

new_vars = {{k: _locals[k] for k in set(_locals) - original_keys}}
print(json.dumps({{"result": result, "new_vars": new_vars}}, default=str))
"""

    # Write the wrapper to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
        tf.write(wrapper)
        script_path = tf.name

    env = "andromeda_code_exec"

    try:
        # Run the temp script inside the conda env
        proc = subprocess.run(
            ["mamba", "run", "-n", env, "python", script_path],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        return payload["result"], payload["new_vars"]
    finally:
        os.remove(script_path)


def eval(code: str, _locals: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    # Store original keys before execution
    original_keys = set(_locals.keys())

    try:
        with contextlib.redirect_stdout(io.StringIO()) as f:
            exec(code, builtins.__dict__, _locals)
        result = f.getvalue()
        if not result:
            result = "<code ran, no output printed to stdout>"
    except Exception as e:
        result = f"Error during execution: {repr(e)}"

    # Determine new variables created during execution
    new_keys = set(_locals.keys()) - original_keys
    new_vars = {key: _locals[key] for key in new_keys}
    return result, new_vars
