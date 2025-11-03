# viz_results.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

strategies = ["hybrid", "majority", "rrf"]
base = Path(".")

# Load and tag results
agg_list, det_list = [], []
for s in strategies:
    agg = pd.read_csv(base / f"evaluation_results_{s}" / "aggregated_results.csv")
    det = pd.read_csv(base / f"evaluation_results_{s}" / "detailed_results.csv")
    agg["strategy"] = s
    det["strategy"] = s
    agg_list.append(agg)
    det_list.append(det)

agg_all = pd.concat(agg_list, ignore_index=True)
det_all = pd.concat(det_list, ignore_index=True)

# 1) Overall comparison (mean across k) per strategy
overall = (agg_all.groupby(["strategy", "System"])
           [["NDCG@k","MAP","MRR","Precision@k","Recall@k","F1@k"]]
           .mean()
           .reset_index())

plt.figure(figsize=(8,4))
sns.barplot(data=overall, x="System", y="NDCG@k", hue="strategy")
plt.title("Mean NDCG (averaged over k) by System and Ground-Truth Strategy")
plt.ylabel("Mean NDCG")
plt.xlabel("")
plt.legend(title="Strategy")
plt.tight_layout()
plt.show()

# 2) Metric vs K (line chart) — NDCG across k
plt.figure(figsize=(8,5))
sns.lineplot(data=agg_all, x="k", y="NDCG@k", hue="System", style="strategy", markers=True)
plt.title("NDCG vs K by System (style = strategy)")
plt.ylabel("NDCG")
plt.tight_layout()
plt.show()

# 3) Per-query robustness (boxplot) at k=10
k_sel = 10
det_k = det_all[det_all["k"] == k_sel].copy()
plt.figure(figsize=(8,5))
sns.boxplot(data=det_k, x="System", y="NDCG@k", hue="strategy")
plt.title(f"Per-query NDCG@{k_sel} Distribution")
plt.ylabel(f"NDCG@{k_sel}")
plt.xlabel("")
plt.tight_layout()
plt.show()

# 4) Precision–Recall scatter by system at k=10 (facet by strategy)
g = sns.FacetGrid(det_k, col="strategy", hue="System", height=4, aspect=1)
g.map_dataframe(sns.scatterplot, x="Precision@k", y="Recall@k", alpha=0.7)
g.add_legend()
g.set_titles(col_template="{col_name}")
for ax in g.axes.flat:
    ax.set_xlim(0,1); ax.set_ylim(0,1)
plt.tight_layout()
plt.show()

# 5) Win counts (system with best NDCG@10 per query)
wins = (det_k.sort_values(["strategy","Query_ID","NDCG@k"], ascending=[True, True, False])
              .groupby(["strategy","Query_ID"]).first().reset_index())
win_counts = wins.groupby(["strategy","System"])["Query_ID"].count().reset_index(name="wins")

plt.figure(figsize=(7,4))
sns.barplot(data=win_counts, x="System", y="wins", hue="strategy")
plt.title(f"Win counts by System (best NDCG@{k_sel} per query)")
plt.ylabel("#Queries won")
plt.xlabel("")
plt.tight_layout()
plt.show()

# 6) Per-query leaderboard heatmap (NDCG@10) per strategy
for s in strategies:
    mat = det_k[det_k["strategy"] == s].pivot(index="Query_ID", columns="System", values="NDCG@k")
    plt.figure(figsize=(7, max(3, 0.4*len(mat))))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f"Per-query NDCG@{k_sel} Heatmap — {s}")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()

# 7) Radar chart helper — optional, one strategy at k=10
# Requires: pip install pandas plotly

import plotly.express as px
metrics = ["Precision@k","Recall@k","F1@k","NDCG@k","MAP","MRR"]
rad = (agg_all[agg_all["k"]==k_sel]
       .groupby(["strategy","System"])[metrics].mean().reset_index())

# Filter one strategy and reshape to long form
rad_h = rad[rad["strategy"]=="hybrid"].drop(columns=["strategy"])
rad_long = rad_h.melt(id_vars=["System"], value_vars=metrics,
                      var_name="metric", value_name="value")

# Optional: handle NaNs (e.g., MRR might be NaN if not available)
rad_long["value"] = rad_long["value"].fillna(0)

fig = px.line_polar(rad_long, r="value", theta="metric",
                    line_close=True, color="System",
                    title=f"Radar — hybrid, k={k_sel}")
fig.show()