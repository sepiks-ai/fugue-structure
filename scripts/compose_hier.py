"""Compose a fugue in three learned layers, each constraining the one below.

    py scripts/compose_hier.py [--ckpt=results/tx4/best.pt] [--key=2_minor] [--voices=3] [--seed=0]
                               [--temperature=1.0] [--subjects=8] [--subject_temperature=0.9] [--ctx=2048]
                               [--out=results/hier/sample] [--cpu=1]

1. Plan (results/plan_model.json, learned from the 48 by scripts/learn_plans.py): the order in which the
   voices enter, then each later entry's voice, scale degree, form (plain or inverted) and distance from the
   one before, all in subject lengths; and where the piece ends.
2. Subject: the token model writes the opening of a fugue, one voice alone after an ENTRY marker, for a
   length drawn from Bach's subjects, several times; the line kept scores best on the model's own confidence
   plus likeness to Bach's 48 subjects (seven measures and the ending, from scripts/learn_subjects.py).
3. Notes: the token model writes the piece. At each entry the plan names, the subject (transposed to the
   planned degree, in an octave that suits the voice, inverted if planned) is written into its voice as
   tokens, exactly as entries appear in training; the model may not run past an entry's start, or write in a
   voice while it states the subject, and writes every other voice itself.

Writes <out>.mid and <out>.json, and prints the recurrence measure (sample_tx.recurrence)."""
import os, sys, json, random, math, importlib.util, torch, mido
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fugue.tx import tokens as T
from fugue.tx.model import Composer
spec = importlib.util.spec_from_file_location("sample_tx", os.path.join(os.path.dirname(__file__), "sample_tx.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)

MAJ, MIN = [0, 2, 4, 5, 7, 9, 11], [0, 2, 3, 5, 7, 8, 10]
SPANS = [(4, 6), (6, 9), (8, 11), (12, 13)]           # subject lengths in crotchets, with their counts in the 48
BAR = 4 * T.TPB                                         # 4/4

def args():
    a = dict(ckpt="results/tx4/best.pt", plan="results/plan_model.json", key="2_minor", voices=3, seed=0, temperature=1.0,
             ctx=2048, out="results/hier/sample", cpu=0, subjects=8, subject_temperature=0.9)
    for x in sys.argv[1:]:
        k, v = x[2:].split("=", 1) if "=" in x else (x[2:], "1"); a[k] = type(a[k])(v)
    return a

def pick(rng, d):
    ks = list(d); u = rng.random() * sum(d[k] for k in ks)
    for k in ks:
        u -= d[k]
        if u <= 0: return k
    return ks[-1]

def sample_plan(PM, nv, rng):
    """The learned plan: as web/learned.js samplePlan, in subject lengths. Voices are register ranks, 0 = top."""
    orders = PM["exposition_order"].get(str(nv), {",".join(map(str, range(nv))): 1})
    full = {k: v for k, v in orders.items() if len(k.split(",")) == nv} or orders
    order = [int(x) for x in pick(rng, full).split(",")]
    L = rng.choice(PM["length"].get(str(nv), PM["length"]["3"])); L = max(nv + 3, min(24, L))
    entries, t = [], 0.0
    for k, r in enumerate(order):
        entries.append({"rank": r, "onset": t, "degree": 4 if k % 2 else 0, "form": "plain"}); t += PM["exposition_gap"]
    prev = entries[-1]
    for _ in range(60):
        pos = min(3, int(4 * t / L))
        if t >= L - 1.5 or (len(entries) > nv + 2 and pick(rng, PM["end"].get(str(pos), {"go": 1, "end": 0})) == "end" and t > 0.6 * L): break
        degree = int(pick(rng, PM["degree"].get(f"{prev['degree']}|{pos}", PM["degree_prior"])))
        gap = max(1.0, min(3, PM["gap_values"][int(pick(rng, PM["gap"][str(pos)]))]))   # no stretto yet
        form = pick(rng, PM["form"][str(pos)]); form = form if form in ("plain", "inv") else "plain"
        move = pick(rng, PM["move"]); rank = prev["rank"] + (-1 if move == "up" else 1 if move == "down" else 0)
        if rank < 0 or rank >= nv: rank = rng.randrange(nv)
        t += gap; prev = {"rank": rank, "onset": t, "degree": degree, "form": form}; entries.append(prev)
    entries.append({"rank": nv - 1, "onset": t + 1.0, "degree": 0, "form": "plain"})   # home: tonic, in the bass
    return order, entries, t + 2.0

class Forcing(S.Grammar):
    """The token grammar, plus the plan: forced token runs at set times, voices held while they state the
    subject, and no running past the next forced event."""
    def __init__(s, nv):
        super().__init__(nv, "4/4"); s.bar = -1; s.forced = []; s.queue = []; s.busy = []; s.solo = None; s.horizon = None; s.limit = None
    def now(s): return s.bar * BAR + max(0, s.pos)
    def next_forced(s): return min((f[0] for f in s.forced), default=None)
    def allowed(s):
        if s.queue: return [s.queue[0]]
        ok = set(super().allowed()); t = s.now()
        ok.discard(T.ID["ENTRY"])                                    # entries come from the plan only
        held = {v for a, b, v in s.busy if a <= t < b} | ({u for u in range(s.nv) if u != s.solo} if s.solo is not None else set())
        ok = {i for i in ok if not (T.VOCAB[i].startswith("VOICE_") and s.prev != "SHIFT" and int(T.VOCAB[i][6:]) in held)}
        tf = s.next_forced(); lim = min(x for x in (tf, s.limit) if x is not None) if (tf is not None or s.limit is not None) else None
        if lim is not None:                                          # no running past the next forced event
            ok = {i for i in ok if not (T.VOCAB[i].startswith("POS_") and s.bar * BAR + int(T.VOCAB[i][4:]) > lim)}
            if (s.bar + 1) * BAR > lim: ok.discard(T.ID["BAR"])
        if tf is not None:
            ok.discard(T.ID["EOS"])
            if s.prev in ("POS", "DUR") and t == tf:                 # lower voices first, then the forced one
                f = min((f for f in s.forced if f[0] == tf), key=lambda f: -f[1])
                ok = {i for i in ok if T.VOCAB[i].startswith("VOICE_") and int(T.VOCAB[i][6:]) > f[1]} | {f[2][0]}
        if s.limit is not None or (s.horizon is not None and t < s.horizon): ok.discard(T.ID["EOS"])
        if s.horizon is not None and t >= s.horizon + BAR and s.prev in ("DUR", "BAR"): ok = {T.ID["EOS"]}
        return sorted(ok)
    def step(s, i):
        tf = s.next_forced()
        if not s.queue and tf is not None and s.now() == tf:
            for f in s.forced:
                if f[0] == tf and f[2][0] == i: s.queue = list(f[2][1:]); s.forced.remove(f); break
        elif s.queue: s.queue.pop(0)
        if i == T.ID["BAR"]: s.bar += 1
        return super().step(i)

@torch.no_grad()
def main():
    a = args(); rng = random.Random(a["seed"]); torch.manual_seed(a["seed"])
    dev = "cuda" if torch.cuda.is_available() and not a["cpu"] else "cpu"
    ck = torch.load(a["ckpt"], map_location=dev); ck["cfg"].setdefault("aux", False)
    model = Composer(**ck["cfg"]).to(dev).eval(); model.load_state_dict(ck["model"])
    PM = json.load(open(a["plan"])); nv = a["voices"]
    tonic, mode = int(a["key"].split("_")[0]), a["key"].split("_")[1]; scale = MIN if mode == "minor" else MAJ
    order, plan, Lsub = sample_plan(PM, nv, rng)
    span = rng.choices([s for s, _ in SPANS], [w for _, w in SPANS])[0] * T.TPB           # subject length, steps
    head = [T.ID["BOS"], T.ID["FORM_FUGUE"], T.ID[f"KEY_{a['key']}"], T.ID["METER_4/4"], T.ID[f"VOICES_{nv}"]]
    def run(st, until, temp):                          # sample until the grammar's time reaches `until` (or EOS)
        seq, aux, g = st["seq"], st["aux"], st["g"]
        while len(seq) < 12000:
            if not g.queue and g.prev in ("DUR",) and g.now() >= until and not g.forced: return
            ok = g.allowed()
            if len(ok) == 1: i = ok[0]
            else:
                w = seq if len(seq) <= a["ctx"] else head + seq[-(a["ctx"] - len(head)):]
                wa = aux if len(aux) <= a["ctx"] else aux[:len(head)] + aux[-(a["ctx"] - len(head)):]
                lg = model(torch.tensor([w], device=dev), torch.tensor([wa], device=dev) if ck["cfg"]["aux"] else None)[0, -1].float()
                mask = torch.full_like(lg, float("-inf")); mask[ok] = 0
                lp = torch.log_softmax(lg + mask, -1)
                i = int(torch.multinomial(torch.softmax((lg + mask) / temp, -1), 1))
                if T.VOCAB[i].startswith(("PITCH_", "INT_")): st["lp"] += float(lp[i]); st["n"] += 1
            seq.append(i); aux.append(g.step(i))
            if i == T.ID["EOS"]: return
            if len(seq) % 500 == 0: print(len(seq), "tokens, bar", g.bar, flush=True)

    # 2. the subject: the plan's first voice, alone, after an entry marker. Several are written; the one kept
    #    scores best on the model's own confidence plus likeness to Bach's 48 subjects (scripts/learn_subjects.py)
    v0 = order[0]; SM = PM["subject"]; triad = {0: "tonic", 3: "third", 4: "third", 7: "fifth"}
    best = None
    for c in range(a["subjects"]):
        st = {"seq": list(head), "aux": [0] * len(head), "g": Forcing(nv), "lp": 0.0, "n": 0}
        st["g"].solo, st["g"].limit = v0, span
        for i in [T.ID["BAR"], T.ID["POS_0"], T.ID["ENTRY"], T.ID["ROLE_x1"], T.ID["SHIFT_0"], T.ID[f"VOICE_{v0}"]]:
            st["seq"].append(i); st["aux"].append(st["g"].step(i))
        run(st, span, a["subject_temperature"])
        sub = [n for n in T.decode(st["seq"])["notes"] if n["v"] == v0 and n["on"] * T.TPB < span]
        sub = [(round(n["on"] * T.TPB), min(round(n["d"] * T.TPB), span - round(n["on"] * T.TPB)), n["m"]) for n in sub]
        if len(sub) < 4: continue
        ms = [m for _, _, m in sub]; iv = [y - x for x, y in zip(ms, ms[1:])]; sp = span / T.TPB
        meas = {"log_density": math.log(len(sub) / sp), "range": max(ms) - min(ms), "steps": sum(abs(x) <= 2 for x in iv) / len(iv),
                "leaps": sum(abs(x) >= 5 for x in iv) / len(iv), "lengths": len({d for _, d, _ in sub}),
                "longest": max(d for _, d, _ in sub) / T.TPB / sp, "peak_at": ms.index(max(ms)) / (len(ms) - 1)}
        like = -0.3 * sum(((meas[k] - mu) / sd) ** 2 for k, (mu, sd) in SM["measures"].items())
        like += math.log(SM["ending"][triad.get((ms[-1] - tonic) % 12, "other")])
        score = st["lp"] / max(1, st["n"]) + like
        print(f"  subject {c}: {len(sub)} notes, {meas['lengths']} lengths, confidence {st['lp'] / max(1, st['n']):.2f}, likeness {like:.2f}")
        if best is None or score > best[0]: best = (score, st, sub)
    _, st, subj = best
    seq, aux, g = st["seq"], st["aux"], st["g"]
    g.solo = None; g.limit = None; g.busy.append((0, span, v0))
    print("subject:", [(m, d) for _, d, m in subj])
    # 3. the plan's later entries, forced into their voices
    R = {r: (lo, hi) for r, lo, hi in PM["ranges"].get(str(nv), PM["ranges"]["3"])}
    rel = [m - subj[0][2] for _, _, m in subj]; mid = (min(rel) + max(rel)) / 2
    entries_out = [{"voice": v0, "onset": 0, "degree": 0, "form": "plain", "shift": 0}]
    for e in plan[1:]:
        t0 = round(e["onset"] * span / T.TPB) * T.TPB; lo, hi = R[e["rank"]]; semis = scale[e["degree"]]
        sign = -1 if e["form"] == "inv" else 1
        k = min(range(-3, 4), key=lambda k: abs(semis + 12 * k + sign * mid - (lo + hi) / 2)); shift = semis + 12 * k
        notes, prev = [], None
        for j, (on, d, m) in enumerate(subj):
            p = subj[0][2] + shift + sign * (m - subj[0][2])
            p = max(T.LO, min(T.HI, p))
            if j == 0: toks = [T.ID["ENTRY"], T.ID[f"ROLE_{'inv' if sign < 0 else 'x1'}"], T.ID[f"SHIFT_{shift % 12}"], T.ID[f"VOICE_{e['rank']}"], T.ID[f"VOICE_{e['rank']}"], T.ID[f"PITCH_{p}"], T.dur_token(d)]
            else: toks = [T.ID[f"VOICE_{e['rank']}"], T.ID[f"INT_{max(-T.MAX_INT, min(T.MAX_INT, p - prev))}"], T.dur_token(d)]
            notes.append((t0 + on, e["rank"], toks)); prev = p
        g.forced += [list(x) for x in notes]; g.busy.append((t0, t0 + span, e["rank"]))
        entries_out.append({"voice": e["rank"], "onset": t0 / T.TPB, "degree": e["degree"], "form": e["form"], "shift": shift})
    end = max(round(e["onset"] * span / T.TPB) * T.TPB + span for e in plan)
    g.horizon = end
    run(st, end + 2 * BAR, a["temperature"])
    if seq[-1] != T.ID["EOS"]: seq.append(T.ID["EOS"])
    piece = T.decode(seq)
    piece["plan"] = {"order": order, "entries": entries_out, "subject_span": span / T.TPB}
    piece["recurrence"] = S.recurrence(piece["notes"])
    os.makedirs(os.path.dirname(a["out"]) or ".", exist_ok=True)
    json.dump({"args": a, "tokens": len(seq), **piece}, open(a["out"] + ".json", "w"))
    mf = mido.MidiFile(ticks_per_beat=480); tr = mido.MidiTrack(); mf.tracks.append(tr); ev = []
    for n in piece["notes"]:
        ev += [(round(n["on"] * 480), 1, n["m"], n["v"]), (round((n["on"] + n["d"]) * 480), 0, n["m"], n["v"])]
    t = 0
    for at, on, m, v in sorted(ev):
        tr.append(mido.Message("note_on" if on else "note_off", note=m, velocity=80 if on else 0, channel=v % 16, time=at - t)); t = at
    mf.save(a["out"] + ".mid")
    print(f"{len(piece['notes'])} notes, {len(entries_out)} entries planned (subject {span / T.TPB:g} crotchets), "
          f"{piece['notes'][-1]['on'] + piece['notes'][-1]['d']:.0f} crotchets, recurrence {piece['recurrence']} (Bach 0.02-0.05)")

if __name__ == "__main__":
    main()
