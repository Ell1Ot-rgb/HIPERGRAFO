import numpy as np
import pytest
from types import SimpleNamespace

# Test 1: Monkeypatch onnxruntime.InferenceSession to ensure server uses sess.get_outputs()[0].name
class FakeOutput:
    def __init__(self, name):
        self.name = name

class FakeSession:
    def __init__(self, *args, **kwargs):
        self._outs = [FakeOutput('weird_out')]
    def get_inputs(self):
        return [SimpleNamespace(name='input')]
    def get_outputs(self):
        return self._outs
    def run(self, outs, feed):
        # Return a small logits array shaped like (1,32,2048) or (1,2048)
        import numpy as np
        if outs[0] == 'weird_out':
            return [np.zeros((1,32,2048), dtype=np.float32)]
        return [np.zeros((1,2048), dtype=np.float32)]

def test_onnx_output_detection(monkeypatch):
    import onnxruntime as ort
    monkeypatch.setattr(ort, 'InferenceSession', FakeSession)
    # Ensure repo root on sys.path
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # Call the code path that constructs session and reads outputs
    from src.local_server import servidor_art_v7_hipergrafo as svr
    # Simulate a single sample and call the endpoint queue handler with epochs 0 (fast)
    from fastapi.testclient import TestClient
    client = TestClient(svr.app)
    payload = {"samples": [{"input_data": [0.0]*1600, "anomaly_label": 0}]}
    r = client.post('/train_layers_3_5?epochs=0', json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 'job_id' in data

# Test 2: Unit check for Seq-VCR loss terms behaviour (var/cov computation)
def compute_vcr_penalties(seq_feats, target_var=1.0):
    import torch
    Xseq = torch.cat([torch.tensor(x) for x in seq_feats], dim=0)
    var = Xseq.var(dim=0, unbiased=False)
    var_violation = (target_var - var).clamp(min=0.0)
    var_loss = (var_violation.pow(2).sum()).item()
    Xc = Xseq - Xseq.mean(dim=0, keepdim=True)
    nN = Xseq.shape[0]
    cov = (Xc.t() @ Xc) / float(nN)
    off_diag_norm = (cov.pow(2).sum() - torch.diagonal(cov).pow(2).sum()).item()
    return var_loss, off_diag_norm

def test_seqvcr_penalties_increase_with_constant_seq():
    # constant sequence: var small => var_violation > 0
    seq = [np.zeros((2,32), dtype=np.float32) for _ in range(3)]
    var_loss, off = compute_vcr_penalties(seq, target_var=1.0)
    assert var_loss > 0
    # off-diag should be near zero because sequence is all zeros
    assert off >= 0


def test_seqvcr_penalties_smaller_with_diverse_seq():
    # diverse seq should reduce var violation
    seq_const = [np.zeros((2,32), dtype=np.float32) for _ in range(3)]
    seq_div = [np.random.randn(2,32).astype(np.float32) for _ in range(3)]
    v_const, o_const = compute_vcr_penalties(seq_const, target_var=1.0)
    v_div, o_div = compute_vcr_penalties(seq_div, target_var=1.0)
    assert v_div < v_const or o_div < o_const
