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

# ───────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
def sweextra_view(url: str = "http://localhost:5000/get_observation", number_of_tests: int = 10):
    """
    Fire a couple of minimal actions against the sandbox-r2e endpoint.
    """

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

    # First action is empty (initialization), second is the actual action
    # actions = ["", action_str]
    # actions = [action_str]

    # R2E dataset entry for both trajectories
    extra_entry =  {'ds': {'FAIL_TO_PASS': ['tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_release_y_specify_type'], 'PASS_TO_PASS': ['tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_added', 'tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_added_missing', 'tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_changed', 'tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_changed_missing', 'tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_current', 'tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_current_missing', 'tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_fixed', 'tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_fixed_missing', 'tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_init', 'tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_release', 'tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_release_missing', 'tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_release_y', 'tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_removed', 'tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_removed_missing', 'tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_suggest', 'tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_suggest_missing', 'tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_suggest_type_fixed', 'tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_suggest_type_removed', 'tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_version_flag', 'tests/integration/test_cli.py::CliIntegrationTestCase::test_cli_view'], '__index_level_0__': 3839, 'base_commit': 'c7b6d92cafd7b261f22ecdccbe36434ed8e268a6', 'created_at': '2021-04-30 04:39:56+00:00', 'environment_setup_commit': 'c7b6d92cafd7b261f22ecdccbe36434ed8e268a6', 'hints_text': '', 'instance_id': 'mc706__changelog-cli-34', 'is_extra_sync': True, 'license': 'mit', 'meta': {'failed_lite_validators': ['has_many_modified_files'], 'has_test_patch': True, 'is_lite': False}, 'patch': 'diff --git a/CHANGELOG.md b/CHANGELOG.md\nindex a4a3db0..eec5d1f 100644\n--- a/CHANGELOG.md\n+++ b/CHANGELOG.md\n@@ -16,6 +16,7 @@ This project adheres to [Semantic Versioning](http://semver.org/) and [Keep a Ch\n   * Existing CHANGELOGs will start using these headers after the new run of `changelog release`\n \n ### Fixed\n+* Fix changelog release --<release type> --yes\n * Format release lines in the same format that keepachangelog.com does\n * Fix Description for pypi\n \ndiff --git a/src/changelog/commands.py b/src/changelog/commands.py\nindex d038747..7b29803 100644\n--- a/src/changelog/commands.py\n+++ b/src/changelog/commands.py\n@@ -60,7 +60,7 @@ def release(release_type: str, auto_confirm: bool) -> None:\n     try:\n         new_version = CL.get_new_release_version(release_type)\n         if auto_confirm:\n-            CL.cut_release()\n+            CL.cut_release(release_type)\n         else:\n             if click.confirm(f"Planning on releasing version {new_version}. Proceed?"):\n                 CL.cut_release(release_type)\n', 'problem_statement': 'The --yes flag on release results in a release type flag being ignored\nIt seems that using the `--yes` flag with release causes the recommended release type to be used rather than one specified by the command. See examples below.\r\n\r\nIt looks like a simple fix - will hopefully raise a PR as soon as I can get my head around how the tests work!\r\n\r\n```\r\n$ changelog suggest\r\n0.7.1\r\n\r\n$ changelog release --patch\r\nPlanning on releasing version 0.7.1. Proceed? [y/N]: N\r\n\r\n$ changelog release --minor\r\nPlanning on releasing version 0.8.0. Proceed? [y/N]: N\r\n\r\n$ changelog release --major\r\nPlanning on releasing version 1.0.0. Proceed? [y/N]: N\r\n\r\n$ changelog release --major --yes\r\n\r\n$ changelog current\r\n0.7.1\r\n```\r\n', 'repo': 'mc706/changelog-cli', 'test_patch': "diff --git a/tests/integration/test_cli.py b/tests/integration/test_cli.py\nindex 9c3e472..9de770d 100644\n--- a/tests/integration/test_cli.py\n+++ b/tests/integration/test_cli.py\n@@ -128,6 +128,15 @@ class CliIntegrationTestCase(unittest.TestCase):\n             suggest = self.runner.invoke(cli, ['current'])\n             self.assertEqual(suggest.output.strip(), '0.1.0')\n \n+    def test_cli_release_y_specify_type(self):\n+        with self.runner.isolated_filesystem():\n+            self.runner.invoke(cli, ['init'])\n+            self.runner.invoke(cli, ['added', 'Adding a new feature'])\n+            result = self.runner.invoke(cli, ['release', '--major', '--yes'])\n+            self.assertTrue(result)\n+            suggest = self.runner.invoke(cli, ['current'])\n+            self.assertEqual(suggest.output.strip(), '1.0.0')\n+\n     def test_cli_release_missing(self):\n         with self.runner.isolated_filesystem():\n             result = self.runner.invoke(cli, ['release'])\n", 'version': '0.0'}, 'id': 'googlefonts__glyphsLib-798_b2a65c3aafdadfc3dd96f52e11143ca98fef6bc2', 'index': 'googlefonts__glyphsLib-798_b2a65c3aafdadfc3dd96f52e11143ca98fef6bc2', 'row_i': 39, 'split': 'train', 'finish': False, 'is_last_step': False}

    from r2egym.agenthub.run.constant import valid_instance_ids
    ds = extra_entry['ds']
    if ds['instance_id'] not in valid_instance_ids:
            print(f"instance_id: {ds['instance_id']} is not in valid_instance_ids") 
            # return None
            raise ValueError(f"instance_id: {ds['instance_id']} is not in valid_instance_ids")
    ds['docker_image'] = f"txharbor.xaminim.com/minimax/algeng/swebench/sweb.eval.x86_64.{ds['instance_id'].lower().replace('__', '_1776_')}:latest"
    extra_entry['ds'] = ds

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
        "sweextra_view": sweextra_view,
    })


if __name__ == "__main__":
    main()
