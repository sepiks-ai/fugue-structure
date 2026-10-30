"""Write a small parity fixture: one fugue's grid, a set of (voice, step, visibility) queries, and the
Python features and logits for each. tests/test_learned.js checks the browser code reproduces them."""
import sys, json, random
import numpy as np, torch
sys.path.insert(0, "src")
from fugue.learn.dataset import load, piece_grid
from fugue.learn.grid import context, bar_info
from fugue.learn.model import Counterpoint
src = sys.argv[1]
m = Counterpoint(); m.load_state_dict(torch.load(src)); m.eval()
f = [x for x in load() if x["piece"] == "wtc2f05"][0]
grid, total = piece_grid(f); bar = bar_info(f["time"]); num = f["time"][0]
rng = random.Random(3); queries = []
for _ in range(60):
    t = rng.randrange(40, total); v = rng.randrange(len(grid)); hide = [rng.random() < 0.5 for _ in grid]
    r = context(grid, v, t, f["tonic_pc"], f["mode"], bar, num, lambda u, s: s < t or not hide[u])
    if r is None: continue
    feats, p0 = r
    with torch.no_grad(): lg = m(torch.tensor([feats])).numpy()[0].tolist()
    queries.append({"v": v, "t": t, "hide": hide, "feats": feats, "p0": p0, "logits": lg})
json.dump({"piece": f["piece"], "tonic": f["tonic_pc"], "mode": f["mode"], "bar": bar, "num": num,
           "grid": [[g[0].tolist(), g[1].astype(int).tolist()] for g in grid], "queries": queries}, open("tests/fixtures/parity.json", "w"))
print(len(queries), "queries")
