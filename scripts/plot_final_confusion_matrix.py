#!/usr/bin/env python3
"""
Plot publication-quality confusion matrix from test results.
Reads results/FINAL_TEST_RESULTS.json.
Output: figures/fig_confusion_matrix_test.pdf

Usage:
  python scripts/plot_final_confusion_matrix.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

# Try test results first, fall back to dev
for results_path in ["results/FINAL_TEST_RESULTS.json",
                     "results/dev_metrics_v2.json"]:
    if os.path.exists(results_path):
        with open(results_path) as f:
            metrics = json.load(f)
        break
else:
    raise FileNotFoundError("No results JSON found")

cm = np.array(metrics['confusion_matrix'])
split = metrics.get('split', 'test')
labels = ['non-CLAGN', 'CLAGN']

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Annotate cells
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > thresh else "black"
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center", color=color, fontsize=18)

ax.set(xticks=[0, 1], yticks=[0, 1],
       xticklabels=labels, yticklabels=labels,
       ylabel='True label', xlabel='Predicted label')
ax.set_title(f'Confusion Matrix ({split} set)\n'
             f'Balanced Acc: {metrics["balanced_accuracy"]:.1%}  '
             f'Recall: {metrics["clagn_recall"]:.1%}',
             fontsize=12)
plt.tight_layout()
plt.savefig(f"figures/fig_confusion_matrix_{split}.pdf",
            dpi=300, bbox_inches='tight')
plt.savefig(f"figures/fig_confusion_matrix_{split}.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved figures/fig_confusion_matrix_{split}.pdf")
