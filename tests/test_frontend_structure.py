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
        diagnostics = index.index('src="diagnostics.js')
        app = index.index('src="app.js')
        self.assertLess(ui, ecology)
        self.assertLess(ecology, diagnostics)
        self.assertLess(diagnostics, app)

    def test_dashboard_uses_shared_ui_helpers(self) -> None:
        helpers = (WEB_ROOT / "ui.js").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn("window.StateNetworkUI = Object.freeze", helpers)
        self.assertIn("} = window.StateNetworkUI", app)
        self.assertNotIn("document.createElementNS(NS", app)

    def test_embodied_training_uses_torch_without_a_backend_selector(self) -> None:
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

        self.assertNotIn('id="embodied-execution-backend"', index)
        self.assertNotIn("Reference Python", index)
        self.assertIn("execution_backend:'torch'", app)
        self.assertNotIn("Legacy comparison controls", index)

    def test_ecology_renderer_reuses_geometry_and_bounds_history(self) -> None:
        ecology = (WEB_ROOT / "ecology.js").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn("class HistoryBuffer", ecology)
        self.assertIn("if (!sameTopology(network, this.topology))", ecology)
        self.assertIn("demoHistory: new HistoryBuffer(360)", app)
        self.assertIn("Math.ceil(tickRate / 30)", app)
        self.assertIn("lastInspectionAt", app)
        self.assertIn("function clearDemoIndividualSelection", app)
        self.assertIn("if (state.demoIndividual) clearDemoIndividualSelection();", app)

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
            "demo-show-rays",
            "demo-show-trajectory",
            "demo-show-info",
            "demo-show-boundary-nodes",
        ):
            self.assertIn(f'id="{element_id}"', index)

    def test_diagnostic_workspace_exposes_robust_loading_and_plot_controls(self) -> None:
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        diagnostics = (WEB_ROOT / "diagnostics.js").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

        for element_id in (
            "diagnostics-load-meta",
            "diagnostics-load-warnings",
            "diagnostics-history-chart",
            "diagnostics-genome-chart",
            "diagnostics-random-chart",
            "diagnostics-sensitivity-chart",
        ):
            self.assertIn(f'id="{element_id}"', index)
        self.assertIn("class DiagnosticLoader", diagnostics)
        self.assertIn("controller?.abort()", diagnostics)
        self.assertIn("waitForJob", app)


if __name__ == "__main__":
    unittest.main()
