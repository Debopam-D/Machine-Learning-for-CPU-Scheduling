"""run_experiment.py -- 120 workloads x 5 policies x 2 repetitions."""
import numpy as np, pandas as pd
from sched_sim import *
rng = np.random.default_rng(2026)
rows, procrows, WL = [], [], []
wid = 0
for n in (4, 8, 12):
    for _ in range(40):
        procs = make_workload(rng, n); WL.append(procs)
        for p in procs:
            procrows.append(dict(wid=wid, n=n, cls=p["cls"], burst1=p["bursts"][0],
                burst2=p["bursts"][1] if len(p["bursts"]) > 1 else 0.0,
                io1=p["io"][0] if len(p["io"]) else 0.0, hint1=p["hints"][0],
                faults=p["faults"], rss=p["rss"], hint_total=p["hints"].sum(),
                arrival=p["arrival"]))
        for pol in POLICIES:
            for rep in range(2):
                m = simulate(procs, pol, np.random.default_rng(10_000 * wid + 100 * rep + POLICIES.index(pol)))
                rows.append(dict(wid=wid, n=n, policy=pol, rep=rep, **m))
        wid += 1
runs = pd.DataFrame(rows); procs_df = pd.DataFrame(procrows)
runs.to_csv("runs_raw.csv", index=False); procs_df.to_csv("processes.csv", index=False)
cv = runs.groupby(["wid", "policy"])["turnaround"].agg(lambda x: x.std(ddof=0) / x.mean())
print("workloads:", wid, "| processes:", len(procs_df), "| timed runs:", len(runs))
print("median CV between reps: %.2f%%  p90: %.2f%%" % (100*cv.median(), 100*cv.quantile(.9)))
