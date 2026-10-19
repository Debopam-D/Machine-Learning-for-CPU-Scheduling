"""sched_sim.py -- workload generator + single-CPU scheduling simulator."""
import heapq
import numpy as np

CLASSES = ["CPU", "IO", "MIXED"]
POLICIES = ["FCFS", "SJF", "RR5", "RR20", "RR80"]
QUANTUM = {"RR5": 5.0, "RR20": 20.0, "RR80": 80.0}
CS_COST = 0.30          # ms charged on every switch between processes
HINT_SIGMA = 0.25       # log-normal noise on runtime hints

def make_process(rng, pid, cls, arrival):
    scale = rng.choice([0.5, 1.0, 1.5])
    if cls == "CPU":
        nb, total, lo, hi = rng.integers(1, 4), rng.uniform(150, 450), 0, 5
    elif cls == "IO":
        nb, total, lo, hi = rng.integers(5, 16), rng.uniform(40, 160), 15, 60
    else:
        nb, total, lo, hi = rng.integers(3, 9), rng.uniform(80, 280), 5, 35
    w = rng.dirichlet(np.ones(nb) * 4)
    bursts = np.maximum(total * scale * w, 1.0)
    io = rng.uniform(lo, hi, nb - 1) * scale if nb > 1 else np.array([])
    hints = bursts * rng.lognormal(0, HINT_SIGMA, nb)
    # synthetic kernel counters (noisy, weakly class dependent)
    faults = max(0, rng.normal({"CPU": 170, "IO": 215, "MIXED": 190}[cls], 70))
    rss = rng.normal({"CPU": 34, "IO": 28, "MIXED": 31}[cls], 12)
    return dict(pid=pid, cls=cls, arrival=arrival, bursts=bursts, io=io,
                hints=hints, faults=faults, rss=max(rss, 4.0))

def make_workload(rng, n):
    span = rng.choice([0, 50, 200, 600])
    arr = np.sort(rng.uniform(0, span, n)) if span else np.zeros(n)
    classes = rng.choice(CLASSES, n)
    return [make_process(rng, i, c, a) for i, (c, a) in enumerate(zip(classes, arr))]

def simulate(procs, policy, rng, jitter=0.01):
    """Event-driven single-CPU simulation. Returns metric dict."""
    q = QUANTUM.get(policy)
    n = len(procs)
    k = [0] * n                                   # current burst index
    rem = [p["bursts"][0] for p in procs]         # remaining time in burst
    first_run, done = [None] * n, [None] * n
    pend = [(p["arrival"], i, i) for i, p in enumerate(procs)]
    heapq.heapify(pend)
    ready, seq = [], n
    t, last, switches, busy = 0.0, None, 0, 0.0
    while pend or ready:
        while pend and pend[0][0] <= t:
            rt, _, i = heapq.heappop(pend); ready.append((rt, i))
        if not ready:
            t = pend[0][0]; continue
        if policy == "SJF":
            hint = lambda x: procs[ready[x][1]]["hints"][k[ready[x][1]]]
            j = min(range(len(ready)), key=lambda x: (hint(x), ready[x][0]))
        else:                                       # FCFS and RR: FIFO order
            j = min(range(len(ready)), key=lambda x: ready[x][0])
        _, i = ready.pop(j)
        if last is not None and last != i:
            t += CS_COST; switches += 1
        last = i
        if first_run[i] is None:
            first_run[i] = t
        run = rem[i] if q is None else min(q, rem[i])
        run *= rng.lognormal(0, jitter)
        t += run; busy += run
        rem[i] -= min(run, rem[i]) if q else rem[i]
        if rem[i] > 1e-9:                           # preempted -> back of queue
            ready.append((t, i)); continue
        k[i] += 1
        if k[i] == len(procs[i]["bursts"]):
            done[i] = t
        else:
            rem[i] = procs[i]["bursts"][k[i]]
            seq += 1
            heapq.heappush(pend, (t + procs[i]["io"][k[i] - 1], seq, i))
    arr = np.array([p["arrival"] for p in procs])
    cpu = np.array([p["bursts"].sum() for p in procs])
    io = np.array([p["io"].sum() for p in procs])
    ta = np.array(done) - arr
    return dict(turnaround=ta.mean(), waiting=(ta - cpu - io).mean(),
                response=(np.array(first_run) - arr).mean(),
                switches=switches, util=busy / t, makespan=t - arr.min())
