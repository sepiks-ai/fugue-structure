"""What Bach's subjects are like: a small statistical portrait of the 48 subjects of the open edition,
used to choose among the subjects the counterpoint model writes. Adds a "subject" section to
results/plan_model.json and web/model/plan_model.json.

For each fugue the subject is the first voice's opening statement (from results/plans_open.json).
Measured: notes per crotchet, range, share of steps, share of leaps of a fourth or more, number of
different note lengths, longest note as a share of the subject, where the highest note falls, and
how the subject ends (tonic, third, fifth or elsewhere). Continuous measures are summarised by mean
and standard deviation (on a log scale where they are rates); the ending by smoothed proportions."""
import json, math, statistics as st

fugues = {f["piece"]: f for f in json.load(open("data/open_wtc.json"))["fugues"]}
plans = json.load(open("results/plans_open.json"))
rows, ends = [], {"tonic": 0.5, "third": 0.5, "fifth": 0.5, "other": 0.5}
for p in plans:
    f = fugues[p["id"]]; e = p["entries"][0]
    # plans number voices in order of entry; the data numbers them top first
    first = min(f["voices"], key=lambda v: v[0][0] if v else 1e9)
    ns = [(o, d, m) for o, d, m in first if e["onset"] - 1e-6 <= o < e["end"] - 1e-6]
    if len(ns) < 3: continue
    span = e["end"] - e["onset"]; ms = [m for _, _, m in ns]; ints = [b - a for a, b in zip(ms, ms[1:])]
    hi = max(ms)
    rows.append({
        "log_density": math.log(len(ns) / span),
        "range": max(ms) - min(ms),
        "steps": sum(abs(x) <= 2 for x in ints) / len(ints),
        "leaps": sum(abs(x) >= 5 for x in ints) / len(ints),
        "lengths": len({round(d, 3) for _, d, _ in ns}),
        "longest": max(d for _, d, _ in ns) / span,
        "peak_at": ms.index(hi) / max(1, len(ms) - 1),
    })
    rel = (ms[-1] - f["tonic_pc"]) % 12
    ends["tonic" if rel == 0 else "third" if rel in (3, 4) else "fifth" if rel == 7 else "other"] += 1
tot = sum(ends.values())
model = {"n": len(rows),
         "measures": {k: [round(st.mean(r[k] for r in rows), 4), round(st.stdev(r[k] for r in rows), 4)] for k in rows[0]},
         "ending": {k: round(v / tot, 4) for k, v in ends.items()}}
print(json.dumps(model, indent=1))
for path in ("results/plan_model.json", "web/model/plan_model.json"):
    pm = json.load(open(path)); pm["subject"] = model
    json.dump(pm, open(path, "w"), separators=(",", ":"))
