"""ml_pipeline.py -- Task A (process type) and Task B (policy selection)."""
import json, time, warnings, numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
SEED = 2026
runs = pd.read_csv("runs_raw.csv"); P = pd.read_csv("processes.csv")
M = runs.groupby(["wid", "n", "policy"]).mean(numeric_only=True).reset_index()
POL = ["FCFS", "SJF", "RR5", "RR20", "RR80"]
def models():
    return {"Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
            "Random Forest": RandomForestClassifier(200, random_state=SEED, n_jobs=1),
            "Logistic Reg.": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            "Majority": DummyClassifier(strategy="most_frequent")}
out = {}

# ---------------- Task A: process-type classification ----------------
FA = ["burst1", "burst2", "io1", "hint1", "faults", "rss"]
XA, yA, gA = P[FA].values, P["cls"].values, P["wid"].values
gkf = GroupKFold(n_splits=5)
resA, oof_proba = {}, None
for name, mdl in models().items():
    pred = cross_val_predict(mdl, XA, yA, cv=gkf, groups=gA)
    resA[name] = dict(acc=accuracy_score(yA, pred), f1=f1_score(yA, pred, average="macro"))
    if name == "Random Forest":
        cm = confusion_matrix(yA, pred, labels=["CPU", "IO", "MIXED"]); rf_pred = pred
        oof_proba = cross_val_predict(mdl, XA, yA, cv=gkf, groups=gA, method="predict_proba")
out["taskA"] = resA; out["confusion_rf"] = cm.tolist()
out["rf_errors_io_mixed"] = int(cm[1, 2] + cm[2, 1]); out["rf_errors_total"] = int(cm.sum() - np.trace(cm))
rf_full = RandomForestClassifier(200, random_state=SEED).fit(XA, yA)
imp = dict(zip(FA, rf_full.feature_importances_.round(3))); out["rf_importance"] = imp
lat = {}
for name, mdl in models().items():
    if name == "Majority": continue
    mdl.fit(XA, yA); x = XA[:1]; t0 = time.perf_counter()
    for _ in range(200): mdl.predict(x)
    lat[name] = (time.perf_counter() - t0) / 200 * 1e6
out["latency_us"] = lat

# ---------------- Task B: workload-level selection ----------------
for k, c in enumerate(["CPU", "IO", "MIXED"]):   # out-of-fold class fractions
    P["pf_" + c] = oof_proba[:, k]
g = P.groupby("wid")
W = pd.DataFrame({"n": g.size(), "f_cpu": g["pf_CPU"].mean(), "f_io": g["pf_IO"].mean(),
    "f_mix": g["pf_MIXED"].mean(), "hint_mean": g["hint_total"].mean(),
    "hint_cv": g["hint_total"].std(ddof=0) / g["hint_total"].mean(),
    "burst1_mean": g["burst1"].mean(), "io1_mean": g["io1"].mean(),
    "span": g["arrival"].max() - g["arrival"].min()}).reset_index()
FB = [c for c in W.columns if c != "wid"]
def cost(lam):
    c = M.assign(cost=M.turnaround + lam * M.response)
    return c.pivot(index="wid", columns="policy", values="cost")[POL]
def best_label(C, cands): return C[cands].idxmin(axis=1)

def rule(row):               # fixed BEFORE data collection, never tuned
    if row.burst1_mean < 40: return "RR20"
    if row.hint_cv > 0.35: return "SJF"
    return "FCFS"

def evaluate(lam, cands, fb=FB):
    C = cost(lam); y = best_label(C, cands).values; X = W[fb].values
    oracle = C[cands].min(axis=1).values
    strat = {}
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    for name, mdl in models().items():
        ratios, acc = np.zeros(len(W)), []
        for tr, te in rkf.split(X, y):
            mdl.fit(X[tr], y[tr]); p = mdl.predict(X[te])
            ratios[te] += [C.loc[W.wid[i], pp] / oracle[i] for i, pp in zip(te, p)]
            acc.append(np.mean(p == y[te]))
        strat[name] = (ratios / 3, np.mean(acc))
    if len(cands) == len(POL):      # rule picks among all five policies
        strat["Rule-based"] = (np.array([C.loc[r.wid, rule(r)] / o for r, o in zip(W.itertuples(), oracle)]),
                               np.mean([rule(r) == l for r, l in zip(W.itertuples(), y)]))
    for p in cands:
        strat["Fixed " + p] = (C[p].values / oracle, np.mean(y == p))
    return strat, y

lams = {"0": 0.0, "0.5": 0.5, "1": 1.0}
resB, store = {}, {}
for key, lam in lams.items():
    s, y = evaluate(lam, POL); store[key] = s
    resB[key] = {k: dict(ratio=float(v[0].mean()), within5=float((v[0] <= 1.05).mean()), label_acc=float(v[1])) for k, v in s.items()}
    if key == "0": out["label_counts"] = pd.Series(y).value_counts().to_dict()
out["taskB"] = resB
sizes = W.n.values
out["by_size"] = {str(n): {k: float(v[0][sizes == n].mean()) for k, v in store["0"].items()} for n in (4, 8, 12)}

# paired statistics (RF vs each rival) with Holm correction
def holm(p):
    p = np.array(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj
rng = np.random.default_rng(SEED); stats = {}
for key in ("0", "1"):
    s = store[key]; rf = s["Random Forest"][0]; rows = []
    for rival in ["Fixed FCFS", "Fixed RR5", "Fixed RR20", "Fixed RR80", "Fixed SJF", "Rule-based", "Majority"]:
        d = s[rival][0] - rf
        try: pv = wilcoxon(d).pvalue
        except ValueError: pv = 1.0
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        rows.append([rival, d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), pv])
    adj = holm([r[4] for r in rows])
    stats[key] = [r[:4] + [float(a)] for r, a in zip(rows, adj)]
out["stats"] = stats

# quantum-only variant
Q = ["RR5", "RR20", "RR80"]; sq, _ = evaluate(0.0, Q); sq1, _ = evaluate(1.0, Q)
out["quantum"] = {"0": {k: float(v[0].mean()) for k, v in sq.items()}, "1": {k: float(v[0].mean()) for k, v in sq1.items()}}
d = sq["Fixed RR20"][0] - sq["Random Forest"][0]
out["quantum_rf_vs_rr20"] = [float(d.mean()), float(wilcoxon(d).pvalue)]
# ablation: remove predicted class fractions
s_ab, _ = evaluate(0.0, POL, fb=[c for c in FB if not c.startswith("f_")])
out["ablation"] = {k: float(s_ab[k][0].mean()) for k in ("Random Forest", "Logistic Reg.", "Decision Tree")}
# feature importance for Task B (RF on turnaround+response labels)
C1 = cost(1.0); rfB = RandomForestClassifier(200, random_state=SEED).fit(W[FB].values, best_label(C1, POL).values)
out["importanceB"] = dict(zip(FB, rfB.feature_importances_.round(3)))
# tree rules
from sklearn.tree import export_text
dt = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(W[FB].values, best_label(C1, POL).values)
out["tree_text"] = export_text(dt, feature_names=FB)
# save arrays for plotting
pd.to_pickle({"store": store, "W": W, "cm": cm, "M": M, "sizes": sizes}, "plotdata.pkl")
json.dump(out, open("results.json", "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in ["taskA", "latency_us", "label_counts", "rf_errors_io_mixed", "rf_errors_total"]}, indent=1, default=float))
for key in ("0", "0.5", "1"):
    print("lam", key); [print("  %-14s %.3f  within5=%.1f%%  labelacc=%.1f%%" % (k, v["ratio"], 100*v["within5"], 100*v["label_acc"])) for k, v in resB[key].items()]
print("by size", json.dumps(out["by_size"], default=float))
for key in ("0", "1"): print("stats", key); [print("  ", r[0], [round(x, 4) for x in r[1:]]) for r in stats[key]]
print("quantum", out["quantum"], out["quantum_rf_vs_rr20"]); print("ablation", out["ablation"]); print(out["importanceB"]); print(out["rf_importance"]); print(out["tree_text"])
