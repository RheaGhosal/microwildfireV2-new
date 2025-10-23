import torch, torch.nn as nn, numpy as np, json
from sklearn.metrics import roc_auc_score
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.model_cnn import build_resnet18
from models.data import make_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TS(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.log_T = nn.Parameter(torch.zeros(()))
    def forward(self, x):
        return self.model(x) / torch.exp(self.log_T)

def collect(loader, model):
    model.eval()
    y, z = [], []
    with torch.no_grad():
        for x, t in loader:
            x = x.to(DEVICE)
            z.append(model(x).squeeze(1).cpu().numpy())
            y.extend(t.numpy().tolist())
    return np.array(y), np.concatenate(z)

def main():
    tr, va, te = make_loaders("data/train.csv", batch_size=64, num_workers=4)
    m = build_resnet18(in_ch=3).to(DEVICE); m.load_state_dict(torch.load("out/best.pt", map_location=DEVICE)); m.eval()
    yv, zv = collect(va, m)
    yt, zt = collect(te, m)

    model_ts = TS(m).to(DEVICE)
    opt = torch.optim.LBFGS([model_ts.log_T], lr=0.1, max_iter=100)

    def loss_closure():
        opt.zero_grad()
        logits = torch.tensor(zv, dtype=torch.float32, device=DEVICE).unsqueeze(1) / torch.exp(model_ts.log_T)
        # BCE on val
        targets = torch.tensor(yv, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        loss = nn.BCEWithLogitsLoss()(logits, targets)
        loss.backward()
        return loss

    opt.step(loss_closure)
    T = float(torch.exp(model_ts.log_T).cpu())
    print(f"Fitted temperature: {T:.3f}")

    # Calibrated test AUROC (should be similar; ECE improves)
    zt_scaled = zt / T
    from sklearn.metrics import roc_auc_score
    print("Test AUROC (calibrated):", roc_auc_score(yt, 1/(1+np.exp(-zt_scaled))))
    with open("out/temperature.json","w") as f:
        json.dump({"T": T}, f, indent=2)

if __name__ == "__main__":
    main()

