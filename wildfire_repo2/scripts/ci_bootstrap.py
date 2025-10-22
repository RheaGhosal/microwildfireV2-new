import json, numpy as np
from sklearn.metrics import roc_auc_score

with open("out/test_probs.json") as f:
    d = json.load(f)
y = np.array(d["y_true"])
p = np.array(d["prob_ens"])  # or "prob_best"

n = len(y); B = 2000
rng = np.random.RandomState(42)
aucs = []
for _ in range(B):
    idx = rng.randint(0, n, n)
    aucs.append(roc_auc_score(y[idx], p[idx]))
aucs = np.array(aucs)
lo, hi = np.percentile(aucs, [2.5, 97.5])
print(f"AUC_bootstrap mean={aucs.mean():.3f}  95% CI=({lo:.3f}, {hi:.3f})")

