"""Optional real GPU test. Never substitutes a mock for official parity."""
import os

import pytest


@pytest.mark.skipif(not os.environ.get("ZSDC_GPU_CONFIG"),reason="GPU/checkpoint/data unavailable; set ZSDC_GPU_CONFIG on server")
def test_official_gpu_parity():
    from zsdc.config import load_config
    from zsdc.evaluation.runner import audit_models
    reports = audit_models(load_config(os.environ["ZSDC_GPU_CONFIG"]))
    assert all(r["passed"] for r in reports), reports
