"""make_figures.py -- all figures and Table 1 from the saved experiment data."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = pd.read_pickle("plotdata.pkl"); R = json.load(open("results.json"))
M, store, cm, sizes = d["M"], d["store"], d["cm"], d["sizes"]
POL = ["FCFS", "SJF", "RR5", "RR20", "RR80"]; C = dict(zip(POL, ["#8a8f98", "#1f6feb", "#e0793a", "#d1a02b", "#9b59b6"]))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
# Table 1 data
t1 = M.pivot_table(index="policy", columns="n", values="turnaround").loc[POL]
oth = M.groupby("policy")[["response", "switches", "util", "makespan"]].mean().loc[POL]
t1 = t1.join(oth); t1.to_csv("table1.csv"); print(t1.round(2))
best = M.loc[M.groupby("wid").turnaround.idxmin()]
wins = best.groupby(["n", "policy"]).size().unstack(fill_value=0); print(wins); print(best.policy.value_counts())
# Fig 1
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
x = np.arange(3); w = .16
for i, p in enumerate(POL): ax[0].bar(x + (i - 2) * w, t1.loc[p, [4, 8, 12]], w, label=p, color=C[p])
ax[0].set_xticks(x, ["n=4", "n=8", "n=12"]); ax[0].set_ylabel("Mean turnaround (ms)"); ax[0].legend(fontsize=7, frameon=False, ncol=2)
ax[1].bar(POL, t1["switches"], color=[C[p] for p in POL]); ax[1].set_ylabel("Context switches / workload")
for i, v in enumerate(t1["switches"]): ax[1].text(i, v + 5, f"{v:.0f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig("fig1.png", dpi=200); plt.close()
# Fig 2
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.25, 1]})
names = ["Decision Tree", "Random Forest", "Logistic Reg.", "Majority"]
acc = [100 * R["taskA"][n]["acc"] for n in names]
b = ax[0].bar(["DT", "RF", "LR", "Majority"], acc, color=["#1f6feb", "#2ea043", "#e0793a", "#8a8f98"])
for r, v in zip(b, acc): ax[0].text(r.get_x() + r.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8)
ax[0].set_ylim(0, 108); ax[0].set_ylabel("Grouped 5-fold accuracy (%)")
im = ax[1].imshow(cm, cmap="Blues"); ax[1].set_xticks(range(3), ["CPU", "I/O", "Mixed"]); ax[1].set_yticks(range(3), ["CPU", "I/O", "Mixed"])
ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("True"); ax[1].spines[:].set_visible(True)
for i in range(3):
    for j in range(3): ax[1].text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout(); plt.savefig("fig2.png", dpi=200); plt.close()
# Fig 3/4 boxplots
def box(key, fn, title):
    order = ["Random Forest", "Logistic Reg.", "Decision Tree", "Rule-based", "Fixed SJF", "Fixed RR5", "Fixed FCFS", "Fixed RR20", "Fixed RR80"]
    lab = ["RF", "LR", "DT", "Rule", "SJF", "RR5", "FCFS", "RR20", "RR80"]
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    bp = ax.boxplot([store[key][o][0] for o in order], tick_labels=lab, showfliers=False, patch_artist=True, medianprops=dict(color="black"))
    for i, p in enumerate(bp["boxes"]): p.set_facecolor("#9ecbff" if i < 3 else ("#d9d9d9" if i == 3 else "#f5c9a8"))
    ax.axhline(1, color="red", lw=.8, ls="--"); ax.set_ylabel("Cost / oracle cost"); ax.set_title(title, fontsize=9)
    plt.tight_layout(); plt.savefig(fn, dpi=200); plt.close()
box("0", "fig3.png", "Objective: average turnaround (lambda = 0)"); box("1", "fig4.png", "Objective: turnaround + 1.0 x response (lambda = 1)")
# Fig 5 by size
fig, ax = plt.subplots(figsize=(7.2, 2.6)); sel = ["Random Forest", "Logistic Reg.", "Rule-based", "Fixed SJF", "Fixed RR5", "Fixed FCFS"]
lab = ["RF", "LR", "Rule", "SJF", "RR5", "FCFS"]; x = np.arange(3); w = .13
for i, (s, l) in enumerate(zip(sel, lab)): ax.bar(x + (i - 2.5) * w, [R["by_size"][str(n)][s] - 1 for n in (4, 8, 12)], w, label=l)
ax.set_xticks(x, ["n=4", "n=8", "n=12"]); ax.set_ylabel("Excess over oracle (ratio - 1)"); ax.legend(ncol=6, fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig("fig5.png", dpi=200); plt.close()
# Fig 6 importance
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
for a, k, t in zip(ax, ["rf_importance", "importanceB"], ["Task A: process type (RF)", "Task B: policy choice (RF, lambda=1)"]):
    s = pd.Series(R[k]).sort_values(); a.barh(s.index, s.values, color="#2ea043"); a.set_title(t, fontsize=9)
plt.tight_layout(); plt.savefig("fig6.png", dpi=200); plt.close()
