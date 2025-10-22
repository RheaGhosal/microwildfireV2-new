import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression

def load(name):
    with open(name) as f:
        d = json.load(f)
    return np.array(d["labels"]), np.array(d["probs"])

out = Path("out")
pairs = {}
for tag, fn in [("CNN", "cnn_test_probs.json"),
                ("Random Forest", "rf_test_probs.json"),
                ("LSTM", "lstm_test_probs.json")]:
    y, p = load(out / fn)
    pairs[tag] = (y, p)

plt.figure(figsize=(7,5))
aucs = {}
for tag, (y, p) in pairs.items():
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    aucs[tag] = auc
    plt.plot(fpr, tpr, label=f"{tag} (AUC = {auc:.3f})")

# conditional fusion (only if CNN & RF labels align perfectly)
y_cnn, p_cnn = pairs["CNN"]
y_rf,  p_rf  = pairs["Random Forest"]
if len(y_cnn)==len(y_rf) and np.array_equal(y_cnn, y_rf):
    X = np.c_[p_cnn, p_rf]
    clf = LogisticRegression(max_iter=1000).fit(X, y_cnn)
    pf = clf.predict_proba(X)[:,1]
    fpr, tpr, _ = roc_curve(y_cnn, pf)
    auc = roc_auc_score(y_cnn, pf)
    aucs["Fusion (stacking)"] = auc
    plt.plot(fpr, tpr, label=f"Fusion (AUC = {auc:.3f})")

plt.plot([0,1],[0,1],"--",linewidth=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison — Micro-Wildfire Prediction")
plt.legend(loc="lower right"); plt.tight_layout()
out.mkdir(exist_ok=True, parents=True)
plt.savefig(out/"roc_curve_comparison.png", dpi=200)
plt.savefig(out/"roc_curve_comparison.pdf")
print("Saved:", out/"roc_curve_comparison.png", "and", out/"roc_curve_comparison.pdf")
for k,v in aucs.items(): print(k, "AUC=", f"{v:.3f}")

