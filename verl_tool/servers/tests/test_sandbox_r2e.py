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
    # logger.info("Payload:\n%s", json.dumps(payload, indent=2))

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
    extra_entry =  {'ds': {'FAIL_TO_PASS': ['tests/classes_test.py::GSPathFromFileTest::test_applyTransform_skew', 'tests/classes_test.py::GSPathFromFileTest::test_applyTransform_translate', 'tests/classes_test.py::GSPathFromFileTest::test_applyTransform_translate_scale'], 'PASS_TO_PASS': ['tests/classes_test.py::FontGlyphsProxyTest::test_remove_glyphs', 'tests/classes_test.py::GSAlignmentZoneFromFileTest::test_attributes', 'tests/classes_test.py::GSAnchorFromFileTest::test_name', 'tests/classes_test.py::GSAnchorFromFileTest::test_position', 'tests/classes_test.py::GSAnchorFromFileTest::test_repr', 'tests/classes_test.py::GSBackgroundLayerTest::test_get_GSBackgroundLayer_background', 'tests/classes_test.py::GSBackgroundLayerTest::test_get_GSBackgroundLayer_foreground', 'tests/classes_test.py::GSBackgroundLayerTest::test_get_GSLayer_background', 'tests/classes_test.py::GSBackgroundLayerTest::test_get_GSLayer_foreground', 'tests/classes_test.py::GSBackgroundLayerTest::test_set_GSBackgroundLayer_background', 'tests/classes_test.py::GSBackgroundLayerTest::test_set_GSBackgroundLayer_foreground', 'tests/classes_test.py::GSBackgroundLayerTest::test_set_GSLayer_background', 'tests/classes_test.py::GSBackgroundLayerTest::test_set_GSLayer_foreground', 'tests/classes_test.py::GSComponentFromFileTest::test_anchor', 'tests/classes_test.py::GSComponentFromFileTest::test_bounds', 'tests/classes_test.py::GSComponentFromFileTest::test_component', 'tests/classes_test.py::GSComponentFromFileTest::test_componentName', 'tests/classes_test.py::GSComponentFromFileTest::test_delete_and_add', 'tests/classes_test.py::GSComponentFromFileTest::test_moreBounds', 'tests/classes_test.py::GSComponentFromFileTest::test_position', 'tests/classes_test.py::GSComponentFromFileTest::test_repr', 'tests/classes_test.py::GSComponentFromFileTest::test_rotation', 'tests/classes_test.py::GSComponentFromFileTest::test_smartComponentValues', 'tests/classes_test.py::GSComponentFromFileTest::test_transform', 'tests/classes_test.py::GSCustomParameterTest::test_plistValue_dict', 'tests/classes_test.py::GSCustomParameterTest::test_plistValue_list', 'tests/classes_test.py::GSCustomParameterTest::test_plistValue_string', 'tests/classes_test.py::GSFontFromFileTest::test_classes', 'tests/classes_test.py::GSFontFromFileTest::test_customParameters', 'tests/classes_test.py::GSFontFromFileTest::test_date', 'tests/classes_test.py::GSFontFromFileTest::test_disableNiceNames', 'tests/classes_test.py::GSFontFromFileTest::test_featurePrefixes', 'tests/classes_test.py::GSFontFromFileTest::test_features', 'tests/classes_test.py::GSFontFromFileTest::test_filepath', 'tests/classes_test.py::GSFontFromFileTest::test_glyphs', 'tests/classes_test.py::GSFontFromFileTest::test_instances', 'tests/classes_test.py::GSFontFromFileTest::test_ints', 'tests/classes_test.py::GSFontFromFileTest::test_kerning', 'tests/classes_test.py::GSFontFromFileTest::test_masters', 'tests/classes_test.py::GSFontFromFileTest::test_note', 'tests/classes_test.py::GSFontFromFileTest::test_pathlike_path', 'tests/classes_test.py::GSFontFromFileTest::test_strings', 'tests/classes_test.py::GSFontFromFileTest::test_userData', 'tests/classes_test.py::GSFontMasterFromFileTest::test_attributes', 'tests/classes_test.py::GSFontMasterFromFileTest::test_default_values', 'tests/classes_test.py::GSFontMasterFromFileTest::test_name', 'tests/classes_test.py::GSFontMasterFromFileTest::test_name_assignment', 'tests/classes_test.py::GSFontTest::test_font_master_proxy', 'tests/classes_test.py::GSFontTest::test_init', 'tests/classes_test.py::GSFontTest::test_repr', 'tests/classes_test.py::GSFontTest::test_update_custom_parameter', 'tests/classes_test.py::GSGlyphFromFileTest::test_color', 'tests/classes_test.py::GSGlyphFromFileTest::test_export', 'tests/classes_test.py::GSGlyphFromFileTest::test_horiz_kerningGroup', 'tests/classes_test.py::GSGlyphFromFileTest::test_horiz_metricsKey', 'tests/classes_test.py::GSGlyphFromFileTest::test_id', 'tests/classes_test.py::GSGlyphFromFileTest::test_layers', 'tests/classes_test.py::GSGlyphFromFileTest::test_layers_missing_master', 'tests/classes_test.py::GSGlyphFromFileTest::test_name', 'tests/classes_test.py::GSGlyphFromFileTest::test_note', 'tests/classes_test.py::GSGlyphFromFileTest::test_parent', 'tests/classes_test.py::GSGlyphFromFileTest::test_smart_component_axes', 'tests/classes_test.py::GSGlyphFromFileTest::test_string', 'tests/classes_test.py::GSGlyphFromFileTest::test_unicode', 'tests/classes_test.py::GSGlyphFromFileTest::test_userData', 'tests/classes_test.py::GSGuideLineTest::test_repr', 'tests/classes_test.py::GSInstanceFromFileTest::test_attributes', 'tests/classes_test.py::GSInstanceFromFileTest::test_default_values', 'tests/classes_test.py::GSLayerFromFileTest::test_anchors', 'tests/classes_test.py::GSLayerFromFileTest::test_annotations', 'tests/classes_test.py::GSLayerFromFileTest::test_background', 'tests/classes_test.py::GSLayerFromFileTest::test_backgroundImage', 'tests/classes_test.py::GSLayerFromFileTest::test_components', 'tests/classes_test.py::GSLayerFromFileTest::test_guides', 'tests/classes_test.py::GSLayerFromFileTest::test_hints', 'tests/classes_test.py::GSLayerFromFileTest::test_hints_from_file', 'tests/classes_test.py::GSLayerFromFileTest::test_leftMetricsKey', 'tests/classes_test.py::GSLayerFromFileTest::test_name', 'tests/classes_test.py::GSLayerFromFileTest::test_parent', 'tests/classes_test.py::GSLayerFromFileTest::test_repr', 'tests/classes_test.py::GSLayerFromFileTest::test_rightMetricsKey', 'tests/classes_test.py::GSLayerFromFileTest::test_smartComponentPoleMapping', 'tests/classes_test.py::GSLayerFromFileTest::test_userData', 'tests/classes_test.py::GSLayerFromFileTest::test_widthMetricsKey', 'tests/classes_test.py::GSNodeFromFileTest::test_index', 'tests/classes_test.py::GSNodeFromFileTest::test_makeNodeFirst', 'tests/classes_test.py::GSNodeFromFileTest::test_name', 'tests/classes_test.py::GSNodeFromFileTest::test_nextNode', 'tests/classes_test.py::GSNodeFromFileTest::test_position', 'tests/classes_test.py::GSNodeFromFileTest::test_prevNode', 'tests/classes_test.py::GSNodeFromFileTest::test_repr', 'tests/classes_test.py::GSNodeFromFileTest::test_smooth', 'tests/classes_test.py::GSNodeFromFileTest::test_toggleConnection', 'tests/classes_test.py::GSNodeFromFileTest::test_type', 'tests/classes_test.py::GSNodeFromFileTest::test_userData', 'tests/classes_test.py::GSPathFromFileTest::test_bounds', 'tests/classes_test.py::GSPathFromFileTest::test_direction', 'tests/classes_test.py::GSPathFromFileTest::test_nodes', 'tests/classes_test.py::GSPathFromFileTest::test_parent', 'tests/classes_test.py::GSPathFromFileTest::test_proxy', 'tests/classes_test.py::GSPathFromFileTest::test_segments', 'tests/classes_test.py::GlyphLayersTest::test_append_layer_same_id', 'tests/classes_test.py::GlyphLayersTest::test_check_master_layer'], 'base_commit': 'b2a65c3aafdadfc3dd96f52e11143ca98fef6bc2^', 'docker_config_path': '/minimax-dialogue/ruobai/cogito_local/r2e-gym/docker_config/sandbox_docker_googlefonts_glyphsLib.json', 'docker_image': 'txharbor.xaminim.com/minimax/algeng/swebench/sweb.eval.x86_64.googlefonts_1776_glyphslib-798:latest', 'edit_patch': 'diff --git a/Lib/glyphsLib/classes.py b/Lib/glyphsLib/classes.py\nindex cc69d47d..87db7b77 100755\n--- a/Lib/glyphsLib/classes.py\n+++ b/Lib/glyphsLib/classes.py\n@@ -1919,20 +1919,11 @@ class GSPath(GSBase):\n \n     # TODO\n     def applyTransform(self, transformationMatrix):\n-        raise NotImplementedError\n-\n-        # Using both skew values (>0.0) produces different results than Glyphs.\n-        # Skewing just on of the two works.\n-        # Needs more attention.\n         assert len(transformationMatrix) == 6\n         for node in self.nodes:\n-            transformation = (\n-                Affine.translation(transformationMatrix[4], transformationMatrix[5])\n-                * Affine.scale(transformationMatrix[0], transformationMatrix[3])\n-                * Affine.shear(\n-                    transformationMatrix[2] * 45.0, transformationMatrix[1] * 45.0\n-                )\n-            )\n+            transformation = Affine(\n+                    transformationMatrix[0], transformationMatrix[1], transformationMatrix[4],\n+                    transformationMatrix[2], transformationMatrix[3], transformationMatrix[5])\n             x, y = (node.position.x, node.position.y) * transformation\n             node.position.x = x\n             node.position.y = y\n', 'instance_id': 'googlefonts__glyphsLib-798_b2a65c3aafdadfc3dd96f52e11143ca98fef6bc2', 'is_extra_sync': True, 'metadata_path': '/minimax-dialogue/ruobai/cogito_local/r2e-gym/buckets/local_repoeval_bucket/repos/googlefonts__glyphsLib-798_b2a65c3aafdadfc3dd96f52e11143ca98fef6bc2/execution_result.json', 'new_commit_id': 'b2a65c3aafdadfc3dd96f52e11143ca98fef6bc2', 'old_commit_id': 'b2a65c3aafdadfc3dd96f52e11143ca98fef6bc2^', 'problem_statement': "Title: Path Transformation Operations Not Working in GSPath\n\nDescription:\nThe `applyTransform` method in GSPath is not functioning, preventing geometric transformations from being applied to paths. When attempting to transform paths using transformation matrices, the operation fails completely.\n\nHere's a minimal example demonstrating the issue:\n\n```python\npath = GSPath()  # A path with some nodes\ntransform_matrix = (1, 0, 0, 1, 50, 25)  # Simple translation matrix\npath.applyTransform(transform_matrix)  # Should translate the path but fails\n```\n\nThe issue affects all types of transformations:\n- Basic translations (moving paths)\n- Scaling operations\n- Skew transformations\n\nFor example, trying to translate a path by (50, 25) units fails when it should move all points in the path by those amounts. Similar issues occur with scaling and skewing operations.\n\nExpected Behavior:\n- Translation should move all points in the path by the specified amounts\n- Scaling should multiply coordinates by the scale factors\n- Skew transformations should properly deform the path according to the transformation matrix\n\nActual Behavior:\nThe method raises NotImplementedError, making it impossible to perform any geometric transformations on paths. This breaks functionality needed for path manipulation and glyph editing operations.", 'repo': 'googlefonts/glyphsLib', 'test_patch': 'diff --git a/tests/classes_test.py b/tests/classes_test.py\nindex b0acd38c..c622b067 100755\n--- a/tests/classes_test.py\n+++ b/tests/classes_test.py\n@@ -1534,6 +1534,35 @@ class GSPathFromFileTest(GSObjectsTestCase):\n     # TODO:\n     # addNodesAtExtremes()\n     # applyTransform()\n+    def test_applyTransform_translate(self):\n+        pathCopy = copy.copy(self.path)\n+        pathCopy.applyTransform((\n+            1, 0, 0, 1, 50, 25\n+        ))\n+        expected = ((402,172), (402,93), (364,32), (262,32))\n+        for i, pt in enumerate(expected):\n+             self.assertEqual(pathCopy.nodes[i].position.x, pt[0])\n+             self.assertEqual(pathCopy.nodes[i].position.y, pt[1])\n+\n+    def test_applyTransform_translate_scale(self):\n+        pathCopy = copy.copy(self.path)\n+        pathCopy.applyTransform((\n+            0.9, 0, 0, 1.2, 50, 25\n+        ))\n+        expected = ((367,201), (367,107), (333,33), (241,33))\n+        for i, pt in enumerate(expected):\n+             self.assertAlmostEqual(pathCopy.nodes[i].position.x, pt[0],0)\n+             self.assertAlmostEqual(pathCopy.nodes[i].position.y, pt[1],0)\n+\n+    def test_applyTransform_skew(self):\n+        pathCopy = copy.copy(self.path)\n+        pathCopy.applyTransform((\n+            1, 0.1, 0.2, 1, 0, 0\n+        ))\n+        expected = ((381,182), (366,103), (315,38), (213,28))\n+        for i, pt in enumerate(expected):\n+             self.assertAlmostEqual(pathCopy.nodes[i].position.x, pt[0],0)\n+             self.assertAlmostEqual(pathCopy.nodes[i].position.y, pt[1],0)\n \n     def test_direction(self):\n         self.assertEqual(self.path.direction, -1)', 'version': '0.0'}, 'id': 'googlefonts__glyphsLib-798_b2a65c3aafdadfc3dd96f52e11143ca98fef6bc2', 'index': 'googlefonts__glyphsLib-798_b2a65c3aafdadfc3dd96f52e11143ca98fef6bc2', 'row_i': 39, 'split': 'train', 'finish': False, 'is_last_step': False}

    # 构建正确的extra_fields结构
    extra_fields = [extra_entry]#, ds_entry]

    # Send request
    _send_test_request(
        url=url,
        trajectory_ids=traj_ids,
        actions=actions,
        extra_fields=extra_fields,
        test_name="r2e-sandbox-Smoke-Test"
    )

    return True

def single_test_reward(url: str = "http://localhost:5000/get_observation",
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
        """<compute_reward>sandbox_r2e</compute_reward>"""
    )

    # First action is empty (initialization), second is the actual action
    # actions = ["", action_str]
    actions = [action_str]

    # R2E dataset entry for both trajectories
    extra_entry =  {'ds': {'FAIL_TO_PASS': ['tests/classes_test.py::GSPathFromFileTest::test_applyTransform_skew', 'tests/classes_test.py::GSPathFromFileTest::test_applyTransform_translate', 'tests/classes_test.py::GSPathFromFileTest::test_applyTransform_translate_scale'], 'PASS_TO_PASS': ['tests/classes_test.py::FontGlyphsProxyTest::test_remove_glyphs', 'tests/classes_test.py::GSAlignmentZoneFromFileTest::test_attributes', 'tests/classes_test.py::GSAnchorFromFileTest::test_name', 'tests/classes_test.py::GSAnchorFromFileTest::test_position', 'tests/classes_test.py::GSAnchorFromFileTest::test_repr', 'tests/classes_test.py::GSBackgroundLayerTest::test_get_GSBackgroundLayer_background', 'tests/classes_test.py::GSBackgroundLayerTest::test_get_GSBackgroundLayer_foreground', 'tests/classes_test.py::GSBackgroundLayerTest::test_get_GSLayer_background', 'tests/classes_test.py::GSBackgroundLayerTest::test_get_GSLayer_foreground', 'tests/classes_test.py::GSBackgroundLayerTest::test_set_GSBackgroundLayer_background', 'tests/classes_test.py::GSBackgroundLayerTest::test_set_GSBackgroundLayer_foreground', 'tests/classes_test.py::GSBackgroundLayerTest::test_set_GSLayer_background', 'tests/classes_test.py::GSBackgroundLayerTest::test_set_GSLayer_foreground', 'tests/classes_test.py::GSComponentFromFileTest::test_anchor', 'tests/classes_test.py::GSComponentFromFileTest::test_bounds', 'tests/classes_test.py::GSComponentFromFileTest::test_component', 'tests/classes_test.py::GSComponentFromFileTest::test_componentName', 'tests/classes_test.py::GSComponentFromFileTest::test_delete_and_add', 'tests/classes_test.py::GSComponentFromFileTest::test_moreBounds', 'tests/classes_test.py::GSComponentFromFileTest::test_position', 'tests/classes_test.py::GSComponentFromFileTest::test_repr', 'tests/classes_test.py::GSComponentFromFileTest::test_rotation', 'tests/classes_test.py::GSComponentFromFileTest::test_smartComponentValues', 'tests/classes_test.py::GSComponentFromFileTest::test_transform', 'tests/classes_test.py::GSCustomParameterTest::test_plistValue_dict', 'tests/classes_test.py::GSCustomParameterTest::test_plistValue_list', 'tests/classes_test.py::GSCustomParameterTest::test_plistValue_string', 'tests/classes_test.py::GSFontFromFileTest::test_classes', 'tests/classes_test.py::GSFontFromFileTest::test_customParameters', 'tests/classes_test.py::GSFontFromFileTest::test_date', 'tests/classes_test.py::GSFontFromFileTest::test_disableNiceNames', 'tests/classes_test.py::GSFontFromFileTest::test_featurePrefixes', 'tests/classes_test.py::GSFontFromFileTest::test_features', 'tests/classes_test.py::GSFontFromFileTest::test_filepath', 'tests/classes_test.py::GSFontFromFileTest::test_glyphs', 'tests/classes_test.py::GSFontFromFileTest::test_instances', 'tests/classes_test.py::GSFontFromFileTest::test_ints', 'tests/classes_test.py::GSFontFromFileTest::test_kerning', 'tests/classes_test.py::GSFontFromFileTest::test_masters', 'tests/classes_test.py::GSFontFromFileTest::test_note', 'tests/classes_test.py::GSFontFromFileTest::test_pathlike_path', 'tests/classes_test.py::GSFontFromFileTest::test_strings', 'tests/classes_test.py::GSFontFromFileTest::test_userData', 'tests/classes_test.py::GSFontMasterFromFileTest::test_attributes', 'tests/classes_test.py::GSFontMasterFromFileTest::test_default_values', 'tests/classes_test.py::GSFontMasterFromFileTest::test_name', 'tests/classes_test.py::GSFontMasterFromFileTest::test_name_assignment', 'tests/classes_test.py::GSFontTest::test_font_master_proxy', 'tests/classes_test.py::GSFontTest::test_init', 'tests/classes_test.py::GSFontTest::test_repr', 'tests/classes_test.py::GSFontTest::test_update_custom_parameter', 'tests/classes_test.py::GSGlyphFromFileTest::test_color', 'tests/classes_test.py::GSGlyphFromFileTest::test_export', 'tests/classes_test.py::GSGlyphFromFileTest::test_horiz_kerningGroup', 'tests/classes_test.py::GSGlyphFromFileTest::test_horiz_metricsKey', 'tests/classes_test.py::GSGlyphFromFileTest::test_id', 'tests/classes_test.py::GSGlyphFromFileTest::test_layers', 'tests/classes_test.py::GSGlyphFromFileTest::test_layers_missing_master', 'tests/classes_test.py::GSGlyphFromFileTest::test_name', 'tests/classes_test.py::GSGlyphFromFileTest::test_note', 'tests/classes_test.py::GSGlyphFromFileTest::test_parent', 'tests/classes_test.py::GSGlyphFromFileTest::test_smart_component_axes', 'tests/classes_test.py::GSGlyphFromFileTest::test_string', 'tests/classes_test.py::GSGlyphFromFileTest::test_unicode', 'tests/classes_test.py::GSGlyphFromFileTest::test_userData', 'tests/classes_test.py::GSGuideLineTest::test_repr', 'tests/classes_test.py::GSInstanceFromFileTest::test_attributes', 'tests/classes_test.py::GSInstanceFromFileTest::test_default_values', 'tests/classes_test.py::GSLayerFromFileTest::test_anchors', 'tests/classes_test.py::GSLayerFromFileTest::test_annotations', 'tests/classes_test.py::GSLayerFromFileTest::test_background', 'tests/classes_test.py::GSLayerFromFileTest::test_backgroundImage', 'tests/classes_test.py::GSLayerFromFileTest::test_components', 'tests/classes_test.py::GSLayerFromFileTest::test_guides', 'tests/classes_test.py::GSLayerFromFileTest::test_hints', 'tests/classes_test.py::GSLayerFromFileTest::test_hints_from_file', 'tests/classes_test.py::GSLayerFromFileTest::test_leftMetricsKey', 'tests/classes_test.py::GSLayerFromFileTest::test_name', 'tests/classes_test.py::GSLayerFromFileTest::test_parent', 'tests/classes_test.py::GSLayerFromFileTest::test_repr', 'tests/classes_test.py::GSLayerFromFileTest::test_rightMetricsKey', 'tests/classes_test.py::GSLayerFromFileTest::test_smartComponentPoleMapping', 'tests/classes_test.py::GSLayerFromFileTest::test_userData', 'tests/classes_test.py::GSLayerFromFileTest::test_widthMetricsKey', 'tests/classes_test.py::GSNodeFromFileTest::test_index', 'tests/classes_test.py::GSNodeFromFileTest::test_makeNodeFirst', 'tests/classes_test.py::GSNodeFromFileTest::test_name', 'tests/classes_test.py::GSNodeFromFileTest::test_nextNode', 'tests/classes_test.py::GSNodeFromFileTest::test_position', 'tests/classes_test.py::GSNodeFromFileTest::test_prevNode', 'tests/classes_test.py::GSNodeFromFileTest::test_repr', 'tests/classes_test.py::GSNodeFromFileTest::test_smooth', 'tests/classes_test.py::GSNodeFromFileTest::test_toggleConnection', 'tests/classes_test.py::GSNodeFromFileTest::test_type', 'tests/classes_test.py::GSNodeFromFileTest::test_userData', 'tests/classes_test.py::GSPathFromFileTest::test_bounds', 'tests/classes_test.py::GSPathFromFileTest::test_direction', 'tests/classes_test.py::GSPathFromFileTest::test_nodes', 'tests/classes_test.py::GSPathFromFileTest::test_parent', 'tests/classes_test.py::GSPathFromFileTest::test_proxy', 'tests/classes_test.py::GSPathFromFileTest::test_segments', 'tests/classes_test.py::GlyphLayersTest::test_append_layer_same_id', 'tests/classes_test.py::GlyphLayersTest::test_check_master_layer'], 'base_commit': 'b2a65c3aafdadfc3dd96f52e11143ca98fef6bc2^', 'docker_config_path': '/minimax-dialogue/ruobai/cogito_local/r2e-gym/docker_config/sandbox_docker_googlefonts_glyphsLib.json', 'docker_image': 'txharbor.xaminim.com/minimax/algeng/swebench/sweb.eval.x86_64.googlefonts_1776_glyphslib-798:latest', 'edit_patch': 'diff --git a/Lib/glyphsLib/classes.py b/Lib/glyphsLib/classes.py\nindex cc69d47d..87db7b77 100755\n--- a/Lib/glyphsLib/classes.py\n+++ b/Lib/glyphsLib/classes.py\n@@ -1919,20 +1919,11 @@ class GSPath(GSBase):\n \n     # TODO\n     def applyTransform(self, transformationMatrix):\n-        raise NotImplementedError\n-\n-        # Using both skew values (>0.0) produces different results than Glyphs.\n-        # Skewing just on of the two works.\n-        # Needs more attention.\n         assert len(transformationMatrix) == 6\n         for node in self.nodes:\n-            transformation = (\n-                Affine.translation(transformationMatrix[4], transformationMatrix[5])\n-                * Affine.scale(transformationMatrix[0], transformationMatrix[3])\n-                * Affine.shear(\n-                    transformationMatrix[2] * 45.0, transformationMatrix[1] * 45.0\n-                )\n-            )\n+            transformation = Affine(\n+                    transformationMatrix[0], transformationMatrix[1], transformationMatrix[4],\n+                    transformationMatrix[2], transformationMatrix[3], transformationMatrix[5])\n             x, y = (node.position.x, node.position.y) * transformation\n             node.position.x = x\n             node.position.y = y\n', 'instance_id': 'googlefonts__glyphsLib-798_b2a65c3aafdadfc3dd96f52e11143ca98fef6bc2', 'is_extra_sync': True, 'metadata_path': '/minimax-dialogue/ruobai/cogito_local/r2e-gym/buckets/local_repoeval_bucket/repos/googlefonts__glyphsLib-798_b2a65c3aafdadfc3dd96f52e11143ca98fef6bc2/execution_result.json', 'new_commit_id': 'b2a65c3aafdadfc3dd96f52e11143ca98fef6bc2', 'old_commit_id': 'b2a65c3aafdadfc3dd96f52e11143ca98fef6bc2^', 'problem_statement': "Title: Path Transformation Operations Not Working in GSPath\n\nDescription:\nThe `applyTransform` method in GSPath is not functioning, preventing geometric transformations from being applied to paths. When attempting to transform paths using transformation matrices, the operation fails completely.\n\nHere's a minimal example demonstrating the issue:\n\n```python\npath = GSPath()  # A path with some nodes\ntransform_matrix = (1, 0, 0, 1, 50, 25)  # Simple translation matrix\npath.applyTransform(transform_matrix)  # Should translate the path but fails\n```\n\nThe issue affects all types of transformations:\n- Basic translations (moving paths)\n- Scaling operations\n- Skew transformations\n\nFor example, trying to translate a path by (50, 25) units fails when it should move all points in the path by those amounts. Similar issues occur with scaling and skewing operations.\n\nExpected Behavior:\n- Translation should move all points in the path by the specified amounts\n- Scaling should multiply coordinates by the scale factors\n- Skew transformations should properly deform the path according to the transformation matrix\n\nActual Behavior:\nThe method raises NotImplementedError, making it impossible to perform any geometric transformations on paths. This breaks functionality needed for path manipulation and glyph editing operations.", 'repo': 'googlefonts/glyphsLib', 'test_patch': 'diff --git a/tests/classes_test.py b/tests/classes_test.py\nindex b0acd38c..c622b067 100755\n--- a/tests/classes_test.py\n+++ b/tests/classes_test.py\n@@ -1534,6 +1534,35 @@ class GSPathFromFileTest(GSObjectsTestCase):\n     # TODO:\n     # addNodesAtExtremes()\n     # applyTransform()\n+    def test_applyTransform_translate(self):\n+        pathCopy = copy.copy(self.path)\n+        pathCopy.applyTransform((\n+            1, 0, 0, 1, 50, 25\n+        ))\n+        expected = ((402,172), (402,93), (364,32), (262,32))\n+        for i, pt in enumerate(expected):\n+             self.assertEqual(pathCopy.nodes[i].position.x, pt[0])\n+             self.assertEqual(pathCopy.nodes[i].position.y, pt[1])\n+\n+    def test_applyTransform_translate_scale(self):\n+        pathCopy = copy.copy(self.path)\n+        pathCopy.applyTransform((\n+            0.9, 0, 0, 1.2, 50, 25\n+        ))\n+        expected = ((367,201), (367,107), (333,33), (241,33))\n+        for i, pt in enumerate(expected):\n+             self.assertAlmostEqual(pathCopy.nodes[i].position.x, pt[0],0)\n+             self.assertAlmostEqual(pathCopy.nodes[i].position.y, pt[1],0)\n+\n+    def test_applyTransform_skew(self):\n+        pathCopy = copy.copy(self.path)\n+        pathCopy.applyTransform((\n+            1, 0.1, 0.2, 1, 0, 0\n+        ))\n+        expected = ((381,182), (366,103), (315,38), (213,28))\n+        for i, pt in enumerate(expected):\n+             self.assertAlmostEqual(pathCopy.nodes[i].position.x, pt[0],0)\n+             self.assertAlmostEqual(pathCopy.nodes[i].position.y, pt[1],0)\n \n     def test_direction(self):\n         self.assertEqual(self.path.direction, -1)', 'version': '0.0'}, 'id': 'googlefonts__glyphsLib-798_b2a65c3aafdadfc3dd96f52e11143ca98fef6bc2', 'index': 'googlefonts__glyphsLib-798_b2a65c3aafdadfc3dd96f52e11143ca98fef6bc2', 'row_i': 39, 'split': 'train', 'finish': False, 'is_last_step': False}

    # 构建正确的extra_fields结构
    extra_fields = [extra_entry]#, ds_entry]

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
        "single_test_reward": single_test_reward,
    })


if __name__ == "__main__":
    main()
