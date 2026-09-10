"""
Deployment Integration Test Suite
=================================
Validates REST API endpoints, response schemas, and LRP conservation laws.
Can be executed with: pytest test_deployment.py OR python test_deployment.py
"""

import os
import sys
import unittest
from fastapi.testclient import TestClient

from app import app, ENGINE


class TestLRPDeployment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize TestClient with lifespan."""
        cls.client = TestClient(app)
        # Trigger startup lifespan manually if TestClient context manager is used
        with TestClient(app) as client:
            cls.client = client

    def test_01_health_check(self):
        """Test GET /api/health endpoint."""
        with TestClient(app) as client:
            res = client.get("/api/health")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["status"], "healthy")
            self.assertTrue(data["models_loaded"]["tabular"])
            self.assertTrue(data["models_loaded"]["image"])
            print("[Pass] Health check: OK")

    def test_02_tabular_metadata(self):
        """Test GET /api/tabular/metadata endpoint."""
        with TestClient(app) as client:
            res = client.get("/api/tabular/metadata")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(len(data["feature_names"]), 30)
            self.assertIn("malignant", data["target_names"])
            self.assertIn("benign", data["target_names"])
            print("[Pass] Tabular metadata: OK")

    def test_03_tabular_samples(self):
        """Test GET /api/tabular/samples endpoint."""
        with TestClient(app) as client:
            res = client.get("/api/tabular/samples")
            self.assertEqual(res.status_code, 200)
            samples = res.json()
            self.assertGreaterEqual(len(samples), 6)
            self.assertEqual(len(samples[0]["raw_features"]), 30)
            print("[Pass] Tabular sample bank: OK")

    def test_04_tabular_explain_lrp0_conservation(self):
        """Test POST /api/tabular/explain with LRP-0 exact conservation."""
        with TestClient(app) as client:
            samples_res = client.get("/api/tabular/samples")
            sample = samples_res.json()[0]

            payload = {
                "features": sample["raw_features"],
                "is_scaled": False,
                "rule": "lrp-0"
            }
            res = client.post("/api/tabular/explain", json=payload)
            self.assertEqual(res.status_code, 200)
            data = res.json()

            self.assertIn(data["predicted_label"], ["malignant", "benign"])
            self.assertEqual(len(data["feature_attributions"]), 30)

            # Strict conservation check: sum(R) == target_logit
            diff = abs(data["target_logit"] - data["sum_input_relevance"])
            self.assertLess(diff, 1e-3, f"Conservation failed: target={data['target_logit']}, sum={data['sum_input_relevance']}")
            self.assertLess(data["conservation_error"], 1e-3)
            print(f"[Pass] Tabular LRP-0 exact conservation: diff={diff:.2e}")

    def test_05_tabular_explain_rules(self):
        """Test LRP-epsilon and LRP-gamma rules on tabular data."""
        with TestClient(app) as client:
            samples_res = client.get("/api/tabular/samples")
            sample = samples_res.json()[0]

            for rule in ["lrp-epsilon", "lrp-gamma"]:
                payload = {
                    "features": sample["raw_features"],
                    "is_scaled": False,
                    "rule": rule,
                    "epsilon": 0.05,
                    "gamma": 0.3
                }
                res = client.post("/api/tabular/explain", json=payload)
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["rule"], rule)
                self.assertEqual(len(data["feature_attributions"]), 30)
            print("[Pass] Tabular LRP-epsilon & LRP-gamma rules: OK")

    def test_06_image_samples(self):
        """Test GET /api/image/samples endpoint."""
        with TestClient(app) as client:
            res = client.get("/api/image/samples")
            self.assertEqual(res.status_code, 200)
            samples = res.json()
            self.assertEqual(len(samples), 10)
            print("[Pass] Image sample bank: OK")

    def test_07_image_explain_and_contrastive(self):
        """Test POST /api/image/explain with contrastive competitor attribution."""
        with TestClient(app) as client:
            payload = {
                "sample_id": 3,
                "rule": "lrp-epsilon",
                "competitor_class": 8
            }
            res = client.post("/api/image/explain", json=payload)
            self.assertEqual(res.status_code, 200)
            data = res.json()

            self.assertEqual(data["target_class"], 3)
            self.assertEqual(len(data["heatmap_8x8"]), 8)
            self.assertEqual(len(data["heatmap_8x8"][0]), 8)

            # Contrastive explanation verification
            self.assertIsNotNone(data["contrastive"])
            self.assertEqual(data["contrastive"]["competitor_class"], 8)
            self.assertEqual(len(data["contrastive"]["competitor_heatmap_8x8"]), 8)
            print("[Pass] Image LRP and contrastive explanation: OK")

    def test_08_error_handling_invalid_input(self):
        """Test error handling for bad inputs."""
        with TestClient(app) as client:
            # Bad feature dimension
            bad_payload = {
                "features": [1.0, 2.0, 3.0], # only 3 features
                "rule": "lrp-0"
            }
            res = client.post("/api/tabular/explain", json=bad_payload)
            self.assertEqual(res.status_code, 400)
            print("[Pass] Error handling on invalid feature dimensions: OK")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLRPDeployment)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
