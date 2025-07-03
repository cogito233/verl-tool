#!/usr/bin/env python
"""
Structured smoke-test for the Text-Browser tool server.

Run the server first, e.g.:
    python -m verl_tool.servers.serve \
        --tool_type sandbox_r2e \
        --url=http://localhost:5000/get_observation

Then execute:
    python -m verl_tool.servers.tests.test_sandbox_r2e single_test \
        --url=http://localhost:5000/get_observation
"""

import json
import uuid
import logging
import requests
import fire
import time
import regex as re
import numpy as np
from typing import List, Dict, Any, Tuple

CONTROL_CHAR_RE = re.compile(
    # this matches U+0000 through U+001F, excluding tab(09), LF(0A), CR(0D)
    r'[\x00-\x08\x0B\x0C\x0E-\x1F]'
)

# ───────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def sanitize_request(obj: Any) -> Any:
    """
    Recursively walk through obj and:
      - For dicts: sanitize each value
      - For lists/tuples: sanitize each element
      - For strings: remove embedded nulls (and other control chars)
      - Leave other types untouched
    """
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, dict):
        return {sanitize_request(key): sanitize_request(val) for key, val in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(sanitize_request(item) for item in obj)
    elif isinstance(obj, str):
        # strip NUL (\x00) and other C0 control chars
        return CONTROL_CHAR_RE.sub('', obj)
    else:
        return obj

# ───────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────
def _send_test_request(url: str,
                       trajectory_ids: list[str],
                       actions: list[str],
                       extra_fields: list[dict],
                       test_name: str):
    """
    Build the payload, POST to the tool server, and pretty-print the response.
    """
    payload = {
        "trajectory_ids": trajectory_ids,
        "actions": actions,
        "extra_fields": extra_fields,
    }

    # logger.info(f"=== {test_name} ===")
    # logger.info("POST %s", url)
    logger.info("Trajectory IDs: %s", trajectory_ids)

    try:
        resp = requests.post(url, json=payload, timeout=900)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Request error: %s", e)
        return {"error": str(e)}

    try:
        data = resp.json()
    except json.JSONDecodeError:
        logger.error("Response is not valid JSON:\n%s", resp.text[:500])
        return {"error": "invalid-json"}

    logger.info("Response:\n%s", json.dumps(data, indent=2))
    return data


# ───────────────────────────────────────────────
# Browser tests
# ───────────────────────────────────────────────
def swe_verified_view(url: str = "http://localhost:5000/get_observation", number_of_tests: int = 10):
    """
    Fire a couple of minimal actions against the sandbox-r2e endpoint.
    """
    import pandas as pd
    df = pd.read_parquet("/minimax-dialogue/users/ruobai/rl_r2e/data/r2e_swe_verified/test.parquet")
    extra_entry = df.iloc[0]['extra_info']
    extra_entry = sanitize_request(extra_entry)

    # Generate two unique trajectory IDs to simulate two parallel agents
    traj_ids = [f"test-r2e-{i}-{uuid.uuid4()}" for i in range(number_of_tests)]

    # Action: simple bash command to test basic functionality
    action_str = (
        """I need to understand the current state of the repository before making changes. Let me start by exploring the file structure to see what files are available.

<function=file_editor>
  <parameter=command>view</parameter>
  <parameter=path>/testbed</parameter>
</function>"""
    )
    actions = [action_str] * number_of_tests

    # 构建正确的extra_fields结构
    extra_fields = [extra_entry] * number_of_tests


    print(f"################### Step1 start at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    # Send request
    _send_test_request(
        url=url,
        trajectory_ids=traj_ids,
        actions=actions,
        extra_fields=extra_fields,
        test_name="r2e-sandbox-Smoke-Test"
    )
    print(f"################### Step1 finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("Sleeping for 10 seconds")
    time.sleep(10)
    print(f"################### Woke up at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("Done sleeping")
    extra_entry['is_last_step'] = True
    extra_fields = [extra_entry] * number_of_tests
    actions = [""] * number_of_tests
    _send_test_request(
        url=url,
        trajectory_ids=traj_ids,
        actions=actions,
        extra_fields=extra_fields,
        test_name="r2e-sandbox-Smoke-Test"
    )
    print(f"################### Step2 finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    return True

# ───────────────────────────────────────────────
# CLI entry-point
# ───────────────────────────────────────────────
def main():
    """
    Expose the test via Fire.

    Example:
        python -m verl_tool.servers.tests.test_text_browser browser \
            --url=http://localhost:5000/get_observation
    """
    fire.Fire({
        "swe_verified_view": swe_verified_view,
    })


if __name__ == "__main__":
    main()
