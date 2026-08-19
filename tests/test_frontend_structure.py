from __future__ import annotations

import re
import unittest
from pathlib import Path


WEB_ROOT = Path(__file__).parents[1] / "src" / "evolvable_state_network" / "web"


class FrontendStructureTests(unittest.TestCase):
    def test_frontend_modules_load_before_the_dashboard_entrypoint(self) -> None:
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        ui = index.index('src="ui.js')
        ecology = index.index('src="ecology.js')
        app = index.index('src="app.js')
        self.assertLess(ui, ecology)
        self.assertLess(ecology, app)

    def test_dashboard_uses_shared_ui_helpers(self) -> None:
        helpers = (WEB_ROOT / "ui.js").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn("window.StateNetworkUI = Object.freeze", helpers)
        self.assertIn("} = window.StateNetworkUI", app)
        self.assertNotIn("document.createElementNS(NS", app)

    def test_ecology_renderer_reuses_geometry_and_bounds_history(self) -> None:
        ecology = (WEB_ROOT / "ecology.js").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn("class HistoryBuffer", ecology)
        self.assertIn("if (!sameTopology(network, this.topology))", ecology)
        self.assertIn("demoHistory: new HistoryBuffer(360)", app)
        self.assertIn("Math.ceil(tickRate / 30)", app)
        self.assertIn("lastInspectionAt", app)

    def test_dashboard_references_unique_existing_element_ids(self) -> None:
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
        ids = re.findall(r'id="([^"]+)"', index)
        references = set(re.findall(r"\$\('([^']+)'\)", app))

        self.assertEqual(len(ids), len(set(ids)), "HTML element IDs must be unique")
        self.assertEqual(references - set(ids), set(), "Every cached UI element must exist")

    def test_ecology_workspace_exposes_readability_controls(self) -> None:
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        for element_id in (
            "demo-edge-threshold",
            "demo-series-count",
            "demo-node-legend",
            "demo-network-summary",
        ):
            self.assertIn(f'id="{element_id}"', index)


if __name__ == "__main__":
    unittest.main()
