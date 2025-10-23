import matplotlib.pyplot as plt
import numpy as np

# AUROC values with 95% confidence intervals
models = ["CNN", "Random Forest", "LSTM", "Fusion (CNN+RF)"]
auc_means = np.array([0.898, 0.894, 0.774, 0.922])
ci_lowers = np.array([0.842, 0.834, 0.691, 0.868])
ci_uppers = np.array([0.944, 0.942, 0.850, 0.944])

# Compute symmetric error bars
yerr = np.vstack([auc_means - ci_lowers, ci_uppers - auc_means])

plt.figure(figsize=(7, 4))
plt.errorbar(models, auc_means, yerr=yerr, fmt='o', capsize=6, 
             color='darkred', ecolor='gray', linewidth=2, markersize=7)
plt.title("Model AUROC Comparison (95% Confidence Intervals)", fontsize=12)
plt.ylabel("AUROC", fontsize=11)
plt.ylim(0.65, 1.0)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("out/model_accuracy_comparison.pdf")
plt.show()

