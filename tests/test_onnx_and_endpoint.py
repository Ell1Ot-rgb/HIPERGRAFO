import time
import os
import sys
from pathlib import Path
import pytest
import numpy as np
import onnx
import onnxruntime as ort
from fastapi.testclient import TestClient

# Ensure repo root is on sys.path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the FastAPI app
from src.local_server import servidor_art_v7_hipergrafo as server

client = TestClient(server.app)


def test_art_onnx_fixed_loads_and_runs():
    path = "models/art_17_fixed.onnx"
    assert os.path.exists(path), f"ONNX fixed model not found at {path}"
    model = onnx.load(path)
    assert any(op.op_type == 'Slice' for op in model.graph.node) == True

    sess = ort.InferenceSession(path)
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name
    res = sess.run([out], {inp: np.zeros((1, 32), dtype=np.int64)})
    assert res[0].shape[1:] == (32, 2048)


@pytest.mark.timeout(30)
def test_train_layers_3_5_endpoint_queues_and_finishes():
    # Prepare a small batch (one sample)
    payload = {
        "samples": [{
            "input_data": [0.0] * 1600,
            "anomaly_label": 0
        }]
    }

    # pass Seq-VCR params to ensure endpoint accepts them
    r = client.post("/train_layers_3_5?epochs=0&vcr_var_weight=25.0&vcr_cov_weight=1.0&spectral_alpha=0.001&kb_weight=0.1&seq_len=2&edge_threshold=0.5", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "queued"
    job_id = data["job_id"]

    # Poll status until done or timeout
    deadline = time.time() + 20
    status = None
    while time.time() < deadline:
        s = client.get(f"/train_layers_3_5/status/{job_id}")
        assert s.status_code == 200
        job = s.json()
        status = job.get("status")
        if status in ("done", "error"):
            break
        time.sleep(1)

    if status == "error":
        pytest.fail(f"Job failed with error: {job.get('error')}")
    assert status == "done", f"Job did not finish successfully, last status: {status}"
    path = job.get("path")
    assert path and os.path.exists(path), f"Exported model not found at reported path: {path}"
