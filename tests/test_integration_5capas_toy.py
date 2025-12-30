import pytest
import numpy as np
import torch
from src.local_server.servidor_art_v7_hipergrafo import Capa3to5

@ pytest.mark.timeout(60)
def test_capa3to5_learns_toy_separation():
    # Deterministic data: normals zeros, anomalies large positive
    np.random.seed(0)
    torch.manual_seed(0)

    N = 20
    normals = np.zeros((10, 2048), dtype=np.float32)
    anomalies = np.ones((10, 2048), dtype=np.float32) * 5.0
    X = np.vstack([normals, anomalies])
    y = np.array([0]*10 + [1]*10, dtype=np.float32)

    device = torch.device('cpu')
    model = Capa3to5(feature_dim=2048, hidden=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()

    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)

    model.train()
    for epoch in range(30):
        optim.zero_grad()
        # normalize per-sample
        feats = (X_t - X_t.mean(dim=1, keepdim=True)) / (X_t.std(dim=1, keepdim=True) + 1e-6)
        preds, _ = model(feats)
        loss = criterion(preds, y_t)
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        feats = (X_t - X_t.mean(dim=1, keepdim=True)) / (X_t.std(dim=1, keepdim=True) + 1e-6)
        preds, _ = model(feats)
        preds = preds.cpu().numpy()

    norm_mean = preds[:10].mean()
    anom_mean = preds[10:].mean()

    assert anom_mean - norm_mean > 0.2, f"Separation too small: {anom_mean} vs {norm_mean}"
