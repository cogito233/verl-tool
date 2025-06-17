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

    logger.info(f"=== {test_name} ===")
    logger.info("POST %s", url)
    logger.info("Payload:\n%s", json.dumps(payload, indent=2))

    try:
        resp = requests.post(url, json=payload, timeout=910)
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
def single_test(url: str = "http://localhost:5000/get_observation",
                 trajectory_id: str = "test-r2e"):
    """
    Fire a couple of minimal actions against the sandbox-r2e endpoint.
    """

    # Generate two unique trajectory IDs to simulate two parallel agents
    traj_ids = [
        f"{trajectory_id}-{uuid.uuid4()}",
        # f"{trajectory_id}-{uuid.uuid4()}"
    ]

    # Action: simple bash command to test basic functionality
    action_str = (
        """I need to understand the current state of the repository before making changes. Let me start by exploring the file structure to see what files are available.

<function=file_editor>
  <parameter=command>view</parameter>
  <parameter=path>/testbed</parameter>
</function>"""
    )

    # First action is empty (initialization), second is the actual action
    # actions = ["", action_str]
    actions = [action_str]

    # R2E dataset entry for both trajectories
    ds_entry = {
        "instance_id": "BelgianBiodiversityPlatform__python-dwca-reader-101_c406b06443a15edb86fa573650c12a3b80eed2a5",
        "repo": "BelgianBiodiversityPlatform/python-dwca-reader",
        "docker_config_path": "/minimax-dialogue/ruobai/cogito_local/r2e-gym/docker_config/sandbox_docker_BelgianBiodiversityPlatform_python-dwca-reader.json",
        "metadata_path": "/minimax-dialogue/ruobai/cogito_local/r2e-gym/buckets/local_repoeval_bucket/repos/BelgianBiodiversityPlatform__python-dwca-reader-101_c406b06443a15edb86fa573650c12a3b80eed2a5/execution_result.json",
        "docker_image": "txharbor.xaminim.com/minimax/algeng/swebench/sweb.eval.x86_64.belgianbiodiversityplatform_1776_python-dwca-reader-101:latest",
        "base_commit": "c406b06443a15edb86fa573650c12a3b80eed2a5^",
        "old_commit_id": "c406b06443a15edb86fa573650c12a3b80eed2a5^",
        "new_commit_id": "c406b06443a15edb86fa573650c12a3b80eed2a5",
        "version": "0.0",
        "FAIL_TO_PASS": [
            "dwca/test/test_descriptors.py::TestDataFileDescriptor::test_short_headers"
        ],
        "PASS_TO_PASS": [
            "dwca/test/test_descriptors.py::TestArchiveDescriptor::test_exposes_coredescriptor",
            "dwca/test/test_descriptors.py::TestArchiveDescriptor::test_exposes_extensions_2ext",
            "dwca/test/test_descriptors.py::TestArchiveDescriptor::test_exposes_extensions_2ext_ignore",
            "dwca/test/test_descriptors.py::TestArchiveDescriptor::test_exposes_extensions_none",
            "dwca/test/test_descriptors.py::TestArchiveDescriptor::test_exposes_extensions_type",
            "dwca/test/test_descriptors.py::TestArchiveDescriptor::test_exposes_metadata_filename",
            "dwca/test/test_descriptors.py::TestDataFileDescriptor::test_content_raw_element_tag",
            "dwca/test/test_descriptors.py::TestDataFileDescriptor::test_exposes_core_terms",
            "dwca/test/test_descriptors.py::TestDataFileDescriptor::test_exposes_core_type",
            "dwca/test/test_descriptors.py::TestDataFileDescriptor::test_exposes_coreid_index_of_extensions",
            "dwca/test/test_descriptors.py::TestDataFileDescriptor::test_exposes_id_index_of_core",
            "dwca/test/test_descriptors.py::TestDataFileDescriptor::test_exposes_raw_element_tag",
            "dwca/test/test_descriptors.py::TestDataFileDescriptor::test_fields",
            "dwca/test/test_descriptors.py::TestDataFileDescriptor::test_file_details",
            "dwca/test/test_descriptors.py::TestDataFileDescriptor::test_headers_defaultvalue",
            "dwca/test/test_descriptors.py::TestDataFileDescriptor::test_headers_simplecases",
            "dwca/test/test_descriptors.py::TestDataFileDescriptor::test_headers_unordered",
            "dwca/test/test_descriptors.py::TestDataFileDescriptor::test_init_from_file",
            "dwca/test/test_descriptors.py::TestDataFileDescriptor::test_lines_to_ignore",
            "dwca/test/test_descriptors.py::TestDataFileDescriptor::test_tell_if_represents_core"
        ],
        "problem_statement": "Title: Need simplified header names without URL prefixes for Darwin Core Archive files\n\nWhen working with Darwin Core Archive (DwC-A) files, the current header names include full URLs which makes them verbose and harder to work with. For example, headers look like:\n\n```python\ndescriptor = DataFileDescriptor.make_from_metafile_section(xml_section)\nprint(descriptor.headers)\n# Output:\n# ['id', 'http://rs.tdwg.org/dwc/terms/scientificName', \n#  'http://rs.tdwg.org/dwc/terms/basisOfRecord',\n#  'http://rs.tdwg.org/dwc/terms/family']\n```\n\nWe need a way to get simplified header names without the URL prefixes for better readability and easier data manipulation.\n\nExpected behavior:\nThe headers should be available in a simplified format that only includes the term name without the URL prefix. For example:\n\n```python\ndescriptor = DataFileDescriptor.make_from_metafile_section(xml_section)\n# Should return:\n# ['id', 'scientificName', 'basisOfRecord', 'family']\n```\n\nCurrent behavior:\nThere is no way to get simplified headers - only full URLs are available through the `headers` property.\n\nThis would be particularly useful when creating CSV files or displaying data where the full URLs are unnecessary and make the output harder to read.",
        "edit_patch": "diff --git a/CHANGES.txt b/CHANGES.txt\nindex f4a6ff5..57d7b6d 100644\n--- a/CHANGES.txt\n+++ b/CHANGES.txt\n@@ -1,3 +1,8 @@\n+Master\n+------\n+\n+- API new: DataFileDescriptor.short_headers\n+\n v0.11.0 (2017-10-10)\n --------------------\n \ndiff --git a/doc/tutorial.rst b/doc/tutorial.rst\nindex 963772d..fd3421a 100644\n--- a/doc/tutorial.rst\n+++ b/doc/tutorial.rst\n@@ -160,10 +160,10 @@ The easiest way to load the core file as a DataFrame, is to read in the file fro\n         path = dwca.absolute_temporary_path('occurrence.txt')\n \n         # read the core as dataframe (No headers available in the csv-file)\n-        core_df = pd.read_csv(path, delimiter=\"\\t\", header=None, parse_dates=True)\n+        core_df = pd.read_csv(path, delimiter=dwca.descriptor.core.fields_terminated_by, header=None, parse_dates=True)\n \n         # Get the header names from the DwCAReader headers\n-        core_df.columns = [term.split(\"/\")[-1] for term in dwca.descriptor.core.headers]\n+        core_df.columns = dwca.descriptor.core.short_headers\n         # All Pandas functionalities are now available on the core_df DataFrame\n \n \n@@ -265,7 +265,7 @@ The result is the core file joined with the extension files. More information ab\n                                 parse_dates=True, chunksize=chunksize):\n \n             # Get the header names from the DwCAReader headers\n-            chunk.columns = [term.split(\"/\")[-1] for term in dwca.descriptor.core.headers]\n+            chunk.columns = dwca.descriptor.core.short_headers\n             chunk['eventDate'] = pd.to_datetime(chunk['eventDate'])\n \n             # Subselect only the records recorded on a sunday\ndiff --git a/dwca/descriptors.py b/dwca/descriptors.py\nindex b45535d..11ee2f0 100644\n--- a/dwca/descriptors.py\n+++ b/dwca/descriptors.py\n@@ -192,7 +192,15 @@ class DataFileDescriptor(object):\n \n     @property\n     def headers(self):\n-        \"\"\"Return a list of (ordered) column names that can be used to create a header line.\"\"\"\n+        \"\"\"A list of (ordered) column names that can be used to create a header line for the data file.\n+\n+        Example::\n+\n+            ['id', 'http://rs.tdwg.org/dwc/terms/scientificName', 'http://rs.tdwg.org/dwc/terms/basisOfRecord',\n+            'http://rs.tdwg.org/dwc/terms/family', 'http://rs.tdwg.org/dwc/terms/locality']\n+\n+        See also :py:attr:`short_headers` if you prefer less verbose headers.\n+        \"\"\"\n         columns = {}\n \n         for f in self.fields:\n@@ -207,6 +215,18 @@ class DataFileDescriptor(object):\n \n         return [columns[f] for f in sorted(columns.keys())]\n \n+    @property\n+    def short_headers(self):\n+        \"\"\"A list of (ordered) column names (short version) that can be used to create a header line for the data file.\n+\n+           Example::\n+\n+                ['id', 'scientificName', 'basisOfRecord', 'family', 'locality']\n+\n+        See also :py:attr:`headers`.\n+        \"\"\"\n+        return [term.split(\"/\")[-1] for term in self.headers]\n+\n     @property\n     def lines_to_ignore(self):\n         \"\"\"Return the number of header lines/lines to ignore in the data file.\"\"\"\n",
        "test_patch": "diff --git a/dwca/test/test_descriptors.py b/dwca/test/test_descriptors.py\nindex 827df70..a807d69 100644\n--- a/dwca/test/test_descriptors.py\n+++ b/dwca/test/test_descriptors.py\n@@ -229,6 +229,28 @@ class TestDataFileDescriptor(unittest.TestCase):\n \n         self.assertEqual(core_descriptor.headers, expected_headers_core)\n \n+    def test_short_headers(self):\n+        metaxml_section = \"\"\"\n+                <core encoding=\"utf-8\" fieldsTerminatedBy=\"\\t\" linesTerminatedBy=\"\\n\" fieldsEnclosedBy=\"\"\n+                ignoreHeaderLines=\"0\" rowType=\"http://rs.tdwg.org/dwc/terms/Occurrence\">\n+                    <files>\n+                        <location>occurrence.txt</location>\n+                    </files>\n+                    <id index=\"0\" />\n+                    <field default=\"Belgium\" term=\"http://rs.tdwg.org/dwc/terms/country\"/>\n+                    <field index=\"1\" term=\"http://rs.tdwg.org/dwc/terms/scientificName\"/>\n+                    <field index=\"2\" term=\"http://rs.tdwg.org/dwc/terms/basisOfRecord\"/>\n+                    <field index=\"3\" term=\"http://rs.tdwg.org/dwc/terms/family\"/>\n+                    <field index=\"4\" term=\"http://rs.tdwg.org/dwc/terms/locality\"/>\n+                </core>\n+                \"\"\"\n+\n+        core_descriptor = DataFileDescriptor.make_from_metafile_section(ET.fromstring(metaxml_section))\n+\n+        expected_short_headers_core = ['id', 'scientificName', 'basisOfRecord', 'family', 'locality']\n+\n+        self.assertEqual(core_descriptor.short_headers, expected_short_headers_core)\n+\n     def test_headers_unordered(self):\n         metaxml_section = \"\"\"\n         <core encoding=\"utf-8\" fieldsTerminatedBy=\"\\t\" linesTerminatedBy=\"\\n\" fieldsEnclosedBy=\"\"",
        "is_extra_sync": True
    }

    # 构建正确的extra_fields结构
    extra_fields = [ds_entry]#, ds_entry]

    # Send request
    _send_test_request(
        url=url,
        trajectory_ids=traj_ids,
        actions=actions,
        extra_fields=extra_fields,
        test_name="r2e-sandbox-Smoke-Test"
    )

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
        "single_test": single_test,
    })


if __name__ == "__main__":
    main()
