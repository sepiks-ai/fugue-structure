"""Learn a generative model of fugue plans from the 48 open fugues: results/plan_model.json.

A plan is the exposition (the order the voices enter, by register) followed by a sequence of entries.
Each later entry is described by:
  degree   the scale degree the subject starts on, relative to the opening (0..6)
  gap      time since the previous entry, in subject lengths (bucketed; < 1 means stretto)
  form     plain, inverted, or augmented/diminished
  move     same voice as the previous entry, a higher voice, or a lower one
and the plan stops with probability END. Each is conditioned on where we are in the piece
(position quartile) and the degree on the previous entry's degree. Counts are smoothed with a
Dirichlet prior (alpha), so the model can produce plans Bach did not write."""
import json, collections, statistics as st

ALPHA = 0.5
GAPS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0]          # bucket upper bounds, in subject lengths
def gap_bucket(g):
    for i, b in enumerate(GAPS):
        if g <= b + 1e-9: return i
    return len(GAPS)
def gap_value(i): return ([0.15] + [(GAPS[k - 1] + GAPS[k]) / 2 for k in range(1, len(GAPS))] + [7.0])[i]

plans = json.load(open("results/plans_open.json"))
counts = {"degree": collections.defaultdict(collections.Counter), "gap": collections.defaultdict(collections.Counter),
          "form": collections.defaultdict(collections.Counter), "move": collections.Counter(), "end": collections.defaultdict(collections.Counter)}
expo = collections.defaultdict(collections.Counter); lengths = collections.defaultdict(list); expo_gap = []
ranges = collections.defaultdict(list); n_used = 0
for p in plans:
    S = p["subject_span"]; nv = p["voices"]; ents = p["entries"]
    if len(ents) < nv: continue
    n_used += 1
    # exposition: the first entry in each voice, in time order; voices named by register rank (0 = top)
    mean = [st.mean(r) for r in p["voice_ranges"]]; rank = {v: sorted(range(nv), key=lambda u: -mean[u]).index(v) for v in range(nv)}
    seen, ex = set(), []
    for e in ents:
        if e["voice"] not in seen: seen.add(e["voice"]); ex.append(e)
        if len(seen) == nv: break
    expo[str(nv)][",".join(str(rank[e["voice"]]) for e in ex)] += 1
    for a, b in zip(ex, ex[1:]): expo_gap.append((b["onset"] - a["onset"]) / S)
    for v in range(nv): ranges[str(nv)].append([rank[v], p["voice_ranges"][v][0] - p["subject_first_midi"], p["voice_ranges"][v][1] - p["subject_first_midi"]])
    lengths[str(nv)].append(p["length"] / S)
    rest = [e for e in ents if e["onset"] > ex[-1]["onset"] + 1e-9]
    prev = ex[-1]; L = p["length"]
    for k, e in enumerate(rest + [None]):
        pos = min(3, int(4 * (e["onset"] if e else L) / L)); pd = prev["dia_shift"] % 7
        counts["end"][str(pos)]["end" if e is None else "go"] += 1
        if e is None: break
        counts["degree"][f"{pd}|{pos}"][str(e["dia_shift"] % 7)] += 1
        counts["gap"][str(pos)][str(gap_bucket((e["onset"] - prev["onset"]) / S))] += 1
        t = e["transform"]; form = "inv" if t.startswith("INV") else "aug" if "AUG" in t else "dim" if "DIM" in t else "plain"
        counts["form"][str(pos)][form] += 1
        counts["move"]["same" if e["voice"] == prev["voice"] else "up" if rank[e["voice"]] < rank[prev["voice"]] else "down"] += 1
        prev = e
def norm(c, keys):
    tot = sum(c.values()) + ALPHA * len(keys)
    return {k: round((c.get(k, 0) + ALPHA) / tot, 4) for k in keys}
model = {
    "source": "learned from the 48 fugues of Kyle Rother's open edition (CC BY 4.0); see scripts/learn_plans.py",
    "alpha": ALPHA, "plans_used": n_used, "gap_values": [gap_value(i) for i in range(len(GAPS) + 1)],
    "exposition_order": {nv: dict(c) for nv, c in expo.items()},
    "exposition_gap": round(st.median(expo_gap), 3),
    "length": {nv: sorted(round(x, 2) for x in v) for nv, v in lengths.items()},
    "ranges": {nv: [[r, round(st.median(x[1] for x in v if x[0] == r)), round(st.median(x[2] for x in v if x[0] == r))] for r in range(int(nv))] for nv, v in ranges.items()},
    "degree": {k: norm(c, [str(d) for d in range(7)]) for k, c in counts["degree"].items()},
    "degree_prior": norm(sum(counts["degree"].values(), collections.Counter()), [str(d) for d in range(7)]),
    "gap": {k: norm(c, [str(i) for i in range(len(GAPS) + 1)]) for k, c in counts["gap"].items()},
    "form": {k: norm(c, ["plain", "inv", "aug", "dim"]) for k, c in counts["form"].items()},
    "move": norm(counts["move"], ["same", "up", "down"]),
    "end": {k: norm(c, ["go", "end"]) for k, c in counts["end"].items()},
}
json.dump(model, open("results/plan_model.json", "w"), indent=1)
print(n_used, "plans used")
print("degrees of later entries (all positions):", {k: v for k, v in model["degree_prior"].items()})
print("gap (middle of piece):", model["gap"].get("2"))
print("forms:", {k: v for k, v in model["form"].items()})
print("move:", model["move"]); print("exposition orders, 3 voices:", model["exposition_order"].get("3")); print("4 voices:", model["exposition_order"].get("4"))
print("length in subject lengths, 3 voices: median", st.median(model["length"]["3"]))
