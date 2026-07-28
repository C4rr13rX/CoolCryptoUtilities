from __future__ import annotations

import json
import tempfile
from pathlib import Path

from django.test import SimpleTestCase

from services.branddozer_lifecycle import detect_preview
from services.branddozer_product_loop import _remove_false_positive_products, _validate_finished_product
from tools.c0d3rV2.tool_registry import ProductArtifactMaterializerTool, ProjectWorkMapperTool


class ProductDeliveryGateTests(SimpleTestCase):
    def test_readme_only_listing_is_unpublished(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "site").mkdir(); (root / "products" / "fake").mkdir(parents=True)
            (root / "products" / "fake" / "README.md").write_text("description only", encoding="utf-8")
            (root / "site" / "products.json").write_text(json.dumps({"products": [{"slug": "fake", "artifact": "../products/fake/README.md"}]}), encoding="utf-8")
            self.assertEqual(_remove_false_positive_products(root), 1)
            self.assertEqual(json.loads((root / "site" / "products.json").read_text())["products"], [])

    def test_spreadsheet_materializer_creates_valid_workbook(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tool = ProductArtifactMaterializerTool(root)
            result = tool.execute({"kind": "spreadsheet", "spec": {"name": "CRM", "summary": "Customer tracker"}})
            self.assertEqual(result["status"], "materialized")
            validation = _validate_finished_product(root, {"required_extensions": [".xlsx", ".csv"]})
            self.assertTrue(validation["passed"])
            self.assertTrue(validation["primary_artifact"].endswith(".xlsx"))

    def test_thin_spreadsheet_cannot_be_sold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from openpyxl import Workbook
            workbook = Workbook(); workbook.active.append(["name", "value"]); workbook.active.append(["one", "1"])
            workbook.save(root / "thin.xlsx")
            (root / "README.md").write_text("Usage instructions " * 10, encoding="utf-8")
            validation = _validate_finished_product(root, {"required_extensions": [".xlsx"]})
            self.assertFalse(validation["passed"])

    def test_mapper_persists_atomic_scoped_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "existing.py").write_text("class InputShape:\n    pass\n", encoding="utf-8")
            mapper = ProjectWorkMapperTool(root)
            state = mapper.execute({"action": "map", "request": "Add a bounded output adapter"})
            self.assertEqual(state["scope"]["allowed_roots"], [str(root.resolve())])
            task = mapper.execute({"action": "next"})["task"]
            self.assertIn("inputs", task); self.assertIn("outputs", task); self.assertIn("acceptance", task)
            self.assertTrue((root / ".c0d3r" / "project-map.json").exists())

    def test_preview_serves_workspace_root_for_product_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "site").mkdir(); (root / "site" / "index.html").write_text("ok", encoding="utf-8")
            preview = detect_preview(root)
            self.assertEqual(preview["cwd"], str(root.resolve()))
            self.assertEqual(preview["entry"], "site/index.html")
