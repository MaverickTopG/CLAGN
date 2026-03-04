# scripts/validate_color_asymmetry.py
"""
Validate that color asymmetry features discriminate CLAGN from contaminants.
Produces:
  - Figure 1: color_asymmetry distribution by class (violin plot)
  - Figure 2: color_flux_corr distribution by class
  - Figure 3: dust_lag_proxy distribution for CLAGN vs blazars
  - Figure 4: 2D scatter color_asymmetry vs color_flux_corr with class overlay
  - Table 1: Mann-Whitney U test p-values for each feature between CLAGN and each class
  - Table 2: ROC-AUC for color asymmetry alone vs existing composite score alone
             vs combined (color asymmetry + composite score)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load color features
feat = pd.read_csv("data/features/color_asymmetry_features.csv")

# Load existing pipeline scores (from your existing evaluate output)
# Expected columns: source_id, composite_score, label
# Adjust path to match your actual score output file
try:
    scores = pd.read_csv("results/benchmark_v3_scores.csv")
    feat = feat.merge(scores[['source_id','composite_score']], on='source_id', how='left')
    has_scores = True
except FileNotFoundError:
    print("WARNING: No composite scores found — skipping ROC comparison")
    has_scores = False

# ── Class setup ───────────────────────────────────────────────────
CLASSES = {
    'CLAGN':              '#2D6BE4',
    'blazar':             '#E44B3F',
    'supernova':          '#F5A623',
    'normal_agn':         '#1DB87A',
    'normal_agn_variable':'#7B3FE4',
}

feat_clean = feat.dropna(subset=['color_asymmetry','color_flux_corr'])
print(f"Sources with valid color features: {len(feat_clean):,}")
print(feat_clean['label'].value_counts().to_string())

# ── Figure 1: Color asymmetry violin plot ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
class_order = ['CLAGN','normal_agn','normal_agn_variable','blazar','supernova']
data_by_class = []
labels_for_plot = []
for cls in class_order:
    sub = feat_clean[feat_clean['label'] == cls]['color_asymmetry'].dropna()
    if len(sub) > 10:
        # Clip extreme values for visualization
        data_by_class.append(np.clip(sub.values, -8, 8))
        labels_for_plot.append(f"{cls}\n(n={len(sub):,})")

parts = ax.violinplot(data_by_class, positions=range(len(data_by_class)),
                      showmedians=True, showextrema=True)
for i, (pc, cls) in enumerate(zip(parts['bodies'], class_order)):
    pc.set_facecolor(CLASSES.get(cls, '#999999'))
    pc.set_alpha(0.7)

ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5,
           label='No asymmetry (symmetric variability)')
ax.axhline(2, color='#2D6BE4', linestyle=':', linewidth=0.8, alpha=0.7,
           label='Turn-on threshold (asymmetry = 2)')
ax.axhline(-2, color='#E44B3F', linestyle=':', linewidth=0.8, alpha=0.7,
           label='Turn-off threshold (asymmetry = -2)')
ax.set_xticks(range(len(labels_for_plot)))
ax.set_xticklabels(labels_for_plot, fontsize=10)
ax.set_ylabel('Color Asymmetry (W1−W2 slope ratio, early/late)', fontsize=11)
ax.set_title('W1−W2 Color Trajectory Asymmetry by Source Class', fontsize=13)
ax.legend(fontsize=9)
ax.set_ylim(-9, 9)
plt.tight_layout()
plt.savefig("figures/fig1_color_asymmetry_violin.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figures/fig1_color_asymmetry_violin.png", dpi=150, bbox_inches='tight')
print("Saved Figure 1")
plt.close()

# ── Figure 2: Color-flux correlation distribution ─────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for cls, color in CLASSES.items():
    sub = feat_clean[feat_clean['label'] == cls]['color_flux_corr'].dropna()
    if len(sub) > 10:
        ax.hist(sub, bins=40, alpha=0.5, color=color,
                label=f"{cls} (n={len(sub):,})", density=True)
ax.axvline(0.25,  color='#2D6BE4', linestyle='--', label='Turn-on threshold (r=0.25)')
ax.axvline(-0.25, color='#E44B3F', linestyle='--', label='Turn-off threshold (r=-0.25)')
ax.set_xlabel('Pearson r: W1 flux vs W1−W2 color', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Flux-Color Correlation by Source Class', fontsize=13)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig2_color_flux_corr.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figures/fig2_color_flux_corr.png", dpi=150, bbox_inches='tight')
print("Saved Figure 2")
plt.close()

# ── Figure 3: Dust lag proxy distribution ─────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for cls in ['CLAGN','blazar','normal_agn']:
    sub = feat_clean[feat_clean['label'] == cls]['dust_lag_proxy'].dropna()
    sub = sub[(sub > -2) & (sub < 5)]
    if len(sub) > 10:
        ax.hist(sub, bins=30, alpha=0.5, color=CLASSES[cls],
                label=f"{cls} (n={len(sub):,})", density=True)
ax.axvspan(0.3, 3.5, alpha=0.08, color='#2D6BE4',
           label='Physical CLAGN dust lag range (0.3–3.5 yr)')
ax.set_xlabel('Dust Lag Proxy (years, W1 flux → W1−W2 color)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('W1-to-Color Cross-Correlation Peak Lag by Class', fontsize=13)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig3_dust_lag.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figures/fig3_dust_lag.png", dpi=150, bbox_inches='tight')
print("Saved Figure 3")
plt.close()

# ── Figure 4: 2D scatter color_asymmetry vs color_flux_corr ───────
fig, ax = plt.subplots(figsize=(9, 7))
for cls, color in CLASSES.items():
    sub = feat_clean[feat_clean['label'] == cls]
    if len(sub) > 5:
        alpha = 0.6 if cls == 'CLAGN' else 0.15
        size  = 40  if cls == 'CLAGN' else 8
        ax.scatter(sub['color_asymmetry'].clip(-8,8),
                   sub['color_flux_corr'],
                   c=color, alpha=alpha, s=size,
                   label=f"{cls} (n={len(sub):,})", zorder=3 if cls=='CLAGN' else 1)

# Draw decision regions
ax.axvline(2,  color='#2D6BE4', linestyle=':', linewidth=1)
ax.axvline(-2, color='#E44B3F', linestyle=':', linewidth=1)
ax.axhline(0.25,  color='#2D6BE4', linestyle=':', linewidth=1)
ax.axhline(-0.25, color='#E44B3F', linestyle=':', linewidth=1)

# Label quadrants
ax.text( 5,  0.7, 'Turn-ON\ncandidate', ha='center', fontsize=9,
         color='#2D6BE4', fontweight='bold')
ax.text(-5, -0.7, 'Turn-OFF\ncandidate', ha='center', fontsize=9,
         color='#E44B3F', fontweight='bold')

ax.set_xlabel('Color Asymmetry (early/late W1−W2 slope ratio)', fontsize=11)
ax.set_ylabel('Flux-Color Correlation (Pearson r)', fontsize=11)
ax.set_title('Color Asymmetry vs Flux-Color Correlation', fontsize=13)
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim(-9, 9)
ax.set_ylim(-1.05, 1.05)
plt.tight_layout()
plt.savefig("figures/fig4_2d_scatter.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figures/fig4_2d_scatter.png", dpi=150, bbox_inches='tight')
print("Saved Figure 4")
plt.close()

# ── Table 1: Mann-Whitney U tests ─────────────────────────────────
features_to_test = ['color_asymmetry','color_flux_corr','dust_lag_proxy','color_range']
comparison_classes = ['blazar','supernova','normal_agn','normal_agn_variable']
clagn_data = feat_clean[feat_clean['label'] == 'CLAGN']

table1_rows = []
for feat_name in features_to_test:
    clagn_vals = clagn_data[feat_name].dropna().values
    for comp_cls in comparison_classes:
        comp_data = feat_clean[feat_clean['label'] == comp_cls][feat_name].dropna().values
        if len(clagn_vals) < 5 or len(comp_data) < 5:
            continue
        u_stat, p_val = stats.mannwhitneyu(clagn_vals, comp_data, alternative='two-sided')
        effect_size = u_stat / (len(clagn_vals) * len(comp_data))  # r = U / n1*n2
        table1_rows.append({
            'feature': feat_name,
            'comparison': f'CLAGN vs {comp_cls}',
            'n_clagn': len(clagn_vals),
            'n_comp':  len(comp_data),
            'U_stat':  round(u_stat, 1),
            'p_value': f"{p_val:.2e}",
            'effect_size_r': round(effect_size, 3),
            'significant': 'YES' if p_val < 0.001 else 'no'
        })

table1 = pd.DataFrame(table1_rows)
table1.to_csv("results/table1_mannwhitney.csv", index=False)
print("\nTable 1: Mann-Whitney U Tests")
print(table1.to_string(index=False))

# ── Table 2: ROC-AUC comparison ───────────────────────────────────
if has_scores:
    feat_eval = feat_clean.dropna(subset=['composite_score','color_asymmetry','color_flux_corr'])
    y = (feat_eval['label'] == 'CLAGN').astype(int)

    # AUC for composite score alone
    auc_composite = roc_auc_score(y, feat_eval['composite_score'])

    # AUC for color asymmetry alone (use abs value — both turn-on and turn-off)
    color_signal = feat_eval['color_asymmetry'].abs() * feat_eval['color_flux_corr'].abs()
    auc_color = roc_auc_score(y, color_signal)

    # AUC for combined (logistic regression)
    X = feat_eval[['composite_score','color_asymmetry','color_flux_corr',
                   'dust_lag_proxy','color_range']].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_scaled, y)
    proba_combined = lr.predict_proba(X_scaled)[:, 1]
    auc_combined = roc_auc_score(y, proba_combined)

    table2 = pd.DataFrame([
        {'method': 'Existing composite score alone',
         'roc_auc': round(auc_composite, 4), 'note': 'Baseline'},
        {'method': 'Color asymmetry features alone (novel)',
         'roc_auc': round(auc_color, 4),     'note': 'This paper'},
        {'method': 'Combined (composite + color asymmetry)',
         'roc_auc': round(auc_combined, 4),  'note': 'Full method'},
    ])
    table2.to_csv("results/table2_roc_auc.csv", index=False)
    print("\nTable 2: ROC-AUC Comparison")
    print(table2.to_string(index=False))

    # ROC curve figure
    fig, ax = plt.subplots(figsize=(7, 7))
    for method_name, scores_arr, color in [
        ('Composite score alone',     feat_eval['composite_score'], '#999999'),
        ('Color asymmetry alone',     color_signal,                 '#F5A623'),
        ('Combined (this paper)',     proba_combined,               '#2D6BE4'),
    ]:
        fpr, tpr, _ = roc_curve(y, scores_arr)
        auc = roc_auc_score(y, scores_arr)
        ax.plot(fpr, tpr, label=f"{method_name} (AUC={auc:.3f})", linewidth=2)
    ax.plot([0,1],[0,1], 'k--', linewidth=0.8)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: Composite vs Color Asymmetry vs Combined', fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("figures/fig5_roc_curves.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("figures/fig5_roc_curves.png", dpi=150, bbox_inches='tight')
    print("Saved Figure 5")
    plt.close()

print("\nValidation complete. Figures saved to figures/")
print("Tables saved to results/")
