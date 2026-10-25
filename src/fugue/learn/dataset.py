"""Training examples from data/open_wtc.json: (features, next-step token) for every voice and step."""
import json, random
import numpy as np
from .grid import quantise, bar_info, context, target, NAMES, STEP

def load(path="data/open_wtc.json"):
    return json.load(open(path))["fugues"]

def piece_grid(f):
    total = int(round(max(o + d for v in f["voices"] for o, d, m in v) / STEP)) + 1
    return [quantise(v, total) for v in f["voices"]], total

def examples(f, rng, p_unknown=0.5):
    """All examples of one fugue. For each target, each other voice's present and future is hidden
    (as it would be if not yet written) with probability p_unknown; fixed material is visible."""
    grid, total = piece_grid(f)
    bar = bar_info(f["time"]); num = f["time"][0]
    X, Y = [], []
    nv = len(grid)
    for t in range(total):
        hide = {u: rng.random() < p_unknown for u in range(nv)}
        for v in range(nv):
            known = lambda u, s: s < t or not hide[u]
            r = context(grid, v, t, f["tonic_pc"], f["mode"], bar, num, known)
            if r is None: continue
            feats, p0 = r
            X.append(feats); Y.append(target(int(grid[v][0][t]), bool(grid[v][1][t]), p0))
    return np.array(X, np.int16), np.array(Y, np.int8)

def build(fugues, seed=0, copies=1):
    rng = random.Random(seed); Xs, Ys = [], []
    for _ in range(copies):
        for f in fugues:
            X, Y = examples(f, rng); Xs.append(X); Ys.append(Y)
    return np.concatenate(Xs), np.concatenate(Ys)
