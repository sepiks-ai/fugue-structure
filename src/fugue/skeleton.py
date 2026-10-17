"""Skeleton fugue: a given subject laid out on a Bach entry plan, with rule-based free counterpoint.

Pipeline (one module per layer later; one file for now):
  1. PLAN     take a WTC fugue's subject-entry grid (voice, onset, pitch level, transform)
  2. REALISE  place transformed copies of the new subject on that grid, time-scaled to the
              new subject's length, nudging entries that clash badly in stretto
  3. HARMONY  choose one chord per half-bar (Viterbi) to fit the fixed entry notes
  4. FILL     beam search over quavers for the free voices: chord tones on beats, passing
              notes off them, no parallel fifths/octaves, stepwise preference, cadence at end

Major keys only for now. Entries at the tonic, fourth or fifth are transposed exactly (real);
others diatonically within the home key (so a C-major subject on A becomes A aeolian).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path

SCALE = [0, 2, 4, 5, 7, 9, 11]
NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# chords as (name, root pc, pcs) relative to the tonic
CHORDS = [
    ("I", 0, {0, 4, 7}), ("ii", 2, {2, 5, 9}), ("iii", 4, {4, 7, 11}), ("IV", 5, {5, 9, 0}),
    ("V", 7, {7, 11, 2}), ("V7", 7, {7, 11, 2, 5}), ("vi", 9, {9, 0, 4}), ("vii", 11, {11, 2, 5}),
    ("V/V", 2, {2, 6, 9}), ("V/vi", 4, {4, 8, 11}), ("V/ii", 9, {9, 1, 4}),
]
GOOD = {("I", "IV"), ("I", "V"), ("I", "vi"), ("I", "ii"), ("ii", "V"), ("ii", "V7"), ("ii", "vii"),
        ("IV", "V"), ("IV", "I"), ("IV", "ii"), ("V", "I"), ("V", "vi"), ("V7", "I"), ("V7", "vi"),
        ("vi", "ii"), ("vi", "IV"), ("vi", "V"), ("iii", "vi"), ("iii", "IV"), ("vii", "I"), ("I", "V/V"),
        ("I", "V/vi"), ("I", "V/ii"), ("IV", "V/V"), ("vi", "V/ii"), ("iii", "V/vi")}
BAD = {("V", "IV"), ("V7", "IV"), ("V", "ii"), ("V7", "ii"), ("vii", "IV")}
SEC_TARGET = {"V/V": ("V", "V7"), "V/vi": ("vi",), "V/ii": ("ii",)}
CONSONANT = {0, 3, 4, 7, 8, 9}


# ---------------------------------------------------------------- subject handling
def parse_note(tok: str):
    """'C4/1', 'F#4/.5', 'Bb3/0.25' -> (midi, dur)"""
    name, dur = tok.split("/")
    pc = NAMES[name[0].upper()]
    rest = name[1:]
    acc = rest.count("#") - rest.count("b")
    octv = int(rest.replace("#", "").replace("b", ""))
    return 12 * (octv + 1) + pc + acc, float(dur)


def parse_subject(text: str):
    out, t = [], 0.0
    for tok in text.split():
        m, d = parse_note(tok)
        out.append((t, d, m))
        t += d
    return out


def to_degree(midi, tonic):
    """midi -> (diatonic index from tonic C-1 base, alteration)"""
    rel = midi - tonic
    octv, pc = divmod(rel, 12)
    for i in range(len(SCALE) - 1, -1, -1):
        if SCALE[i] <= pc:
            return octv * 7 + i, pc - SCALE[i]
    raise ValueError


def from_degree(deg, alt, tonic):
    octv, i = divmod(deg, 7)
    return tonic + 12 * octv + SCALE[i] + alt


def transform(subject, ev, tonic):
    """Subject notes [(t, d, m)] -> placed notes for template event ev (relative times)."""
    tr = ev["transform"]
    sc = 2.0 if "AUG" in tr else 0.5 if "DIM" in tr else 1.0
    inv = tr.startswith("INV")
    first = subject[0][2]
    d0, _ = to_degree(first, tonic)
    real = ev["midi_shift"] % 12 in (0, 5, 7) and not inv
    out = []
    for t, d, m in subject:
        if real:
            p = m + ev["midi_shift"]
        else:
            deg, alt = to_degree(m, tonic)
            if inv:
                deg = 2 * d0 - deg
            p = from_degree(deg + ev["dia_shift"], alt, tonic)
        out.append((t * sc, d * sc, p))
    return out


# ---------------------------------------------------------------- 2. realise the plan
def dissonance_on_beats(a, b):
    """Count strong-beat dissonances between two placed note lists [(on, dur, midi)]."""
    def at(notes, t):
        for on, d, m in notes:
            if on <= t < on + d:
                return m
        return None
    lo = max(a[0][0], b[0][0])
    hi = min(a[-1][0] + a[-1][1], b[-1][0] + b[-1][1])
    n, t = 0, math.ceil(lo)
    while t < hi:
        x, y = at(a, t), at(b, t)
        if x is not None and y is not None and (abs(x - y) % 12) not in CONSONANT | {5}:
            n += 1
        t += 1.0
    return n


def realise(subject, template, tonic, max_clash=4):
    subj_len = subject[-1][0] + subject[-1][1]
    tsub = template["subject"]
    bach_len = tsub[-1][0] + tsub[-1][1]
    scale = subj_len / bach_len
    evs = sorted((e for e in template["events"] if e["material"] == "SUBJECT"), key=lambda e: e["onset"])
    t0 = evs[0]["onset"]
    placed = []                                  # dicts: voice, onset, notes, ev
    for ev in evs:
        base = round((ev["onset"] - t0) * scale)
        notes_rel = transform(subject, ev, tonic)
        best = None
        for off in (0, 1, -1, 2, -2, 3, -3, 4):
            on = base + off
            if on < 0:
                continue
            notes = [(on + t, d, m) for t, d, m in notes_rel]
            end = notes[-1][0] + notes[-1][1]
            if any(p["voice"] == ev["voice"] and on < p["end"] and p["onset"] < end for p in placed):
                continue
            clash = sum(dissonance_on_beats(notes, p["notes"]) for p in placed
                        if on < p["end"] and p["onset"] < end)
            score = clash + 0.3 * abs(off)
            if best is None or score < best[0]:
                best = (score, clash, on, notes, end)
        if best is None or best[1] > max_clash:
            continue                             # drop an entry that cannot be fitted
        _, clash, on, notes, end = best
        placed.append({"voice": ev["voice"], "onset": on, "end": end, "notes": notes,
                       "transform": ev["transform"], "dia_shift": ev["dia_shift"],
                       "midi_shift": ev["midi_shift"], "clash": clash})
    return placed


# ---------------------------------------------------------------- 3. harmony
def choose_chords(fixed_at, n_slots, steps_per_slot, tonic):
    """Viterbi over half-bar slots. fixed_at: list over quaver steps of list of fixed midi."""
    names = [c[0] for c in CHORDS]

    def emit(slot, c):
        _, root, pcs = c
        s = 0.0
        lowest = None
        for k in range(steps_per_slot):
            t = slot * steps_per_slot + k
            if t >= len(fixed_at):
                break
            w = 1.0 if k % 2 == 0 else 0.4
            if k == 0:
                w = 1.5
            for m in fixed_at[t]:
                pc = (m - tonic) % 12
                s += w if pc in pcs else (-1.5 * w if k % 2 == 0 else -0.3 * w)
                lowest = m if lowest is None or m < lowest else lowest
        if lowest is not None and (lowest - tonic) % 12 == root:
            s += 0.5
        if c[0].startswith("V/"):
            s -= 0.3
        return s

    def trans(a, b):
        if a == b:
            return -0.3
        if a in SEC_TARGET:
            return 2.0 if b in SEC_TARGET[a] else -3.0
        return 1.0 if (a, b) in GOOD else -1.5 if (a, b) in BAD else 0.0

    forced = {n_slots - 1: "I", n_slots - 2: "I", n_slots - 3: "V7"}
    V = [{} for _ in range(n_slots)]
    for i in range(n_slots):
        for c in CHORDS:
            if i in forced and c[0] != forced[i]:
                continue
            e = emit(i, c)
            if i == 0:
                V[i][c[0]] = (e + (2 if c[0] == "I" else 0), None)
                continue
            prev = max(V[i - 1].items(), key=lambda kv: kv[1][0] + trans(kv[0], c[0]))
            V[i][c[0]] = (prev[1][0] + trans(prev[0], c[0]) + e, prev[0])
    seq = [max(V[-1].items(), key=lambda kv: kv[1][0])[0]]
    for i in range(n_slots - 1, 0, -1):
        seq.append(V[i][seq[-1]][1])
    seq.reverse()
    by = {c[0]: c for c in CHORDS}
    return [by[s] for s in seq]


# ---------------------------------------------------------------- 4. fill
def fill(placed, template, tonic, beam=40, per_voice=6, seed=0, step=0.5):
    rng = random.Random(seed)
    nv = template["voices"]
    ranges = [(lo - 2, hi + 2) for lo, hi in template["voice_ranges"]]
    order = sorted(range(nv), key=lambda v: -(ranges[v][0] + ranges[v][1]))   # high to low
    rank = {v: r for r, v in enumerate(order)}
    last_end = max(p["end"] for p in placed)
    total = (math.ceil((last_end + 2) / 4) + 1) * 4            # whole bars, final bar free
    T = int(total / step)
    spb = int(1 / step)                                        # steps per crotchet

    fixed = [[None] * T for _ in range(nv)]
    for p in placed:
        for on, d, m in p["notes"]:
            a, b = int(round(on / step)), int(math.ceil((on + d) / step - 1e-9))
            for t in range(a, min(b, T)):
                if fixed[p["voice"]][t] is None:
                    fixed[p["voice"]][t] = m
    entered = [min(int(round(p["onset"] / step)) for p in placed if p["voice"] == v)
               if any(p["voice"] == v for p in placed) else T for v in range(nv)]
    nxt = [[None] * T for _ in range(nv)]                      # next fixed (step, pitch) per voice
    for v in range(nv):
        cur = None
        for t in range(T - 1, -1, -1):
            if fixed[v][t] is not None and (t == 0 or fixed[v][t - 1] is None):
                cur = (t, fixed[v][t])
            nxt[v][t] = cur
    fixed_at = [[fixed[v][t] for v in range(nv) if fixed[v][t] is not None] for t in range(T)]
    slot_steps = 2 * spb
    n_slots = T // slot_steps
    chords = choose_chords(fixed_at, n_slots, slot_steps, tonic)
    scale_pcs = {(tonic + s) % 12 for s in SCALE}

    def melodic(p, q, qq, hold):
        if q is None:
            return 0.0
        d = abs(p - q)
        c = {0: 0.4 * (hold + 1), 1: 0, 2: 0, 3: 0.6, 4: 0.6, 5: 1.2, 6: 4, 7: 1.6, 8: 2.5, 9: 2.5, 12: 2.5}.get(d, 5)
        if qq is not None and abs(q - qq) >= 5 and not (0 < abs(p - q) <= 2 and (p - q) * (q - qq) < 0):
            c += 1.5
        return c

    # beam state: (cost, cur pitches, prev pitches, holds, parent)
    states = [(0.0, (None,) * nv, (None,) * nv, (0,) * nv, None)]
    final_bar = T - 4 * spb
    for t in range(T):
        _, root, pcs = chords[min(t // slot_steps, n_slots - 1)]
        pcs = {(tonic + x) % 12 for x in pcs}
        root = (tonic + root) % 12
        strong = t % spb == 0
        new = []
        for cost, cur, prev, holds, parent in states:
            options = []
            for v in range(nv):
                if t < entered[v]:
                    options.append([(None, 0.0)])
                    continue
                if fixed[v][t] is not None:
                    options.append([(fixed[v][t], 0.0)])
                    continue
                q, qq = cur[v], prev[v]
                lo, hi = ranges[v]
                base = q if q is not None else (lo + hi) // 2
                cands = []
                for p in range(max(lo, base - 9), min(hi, base + 9) + 1):
                    pc = p % 12
                    if pc not in scale_pcs and pc not in pcs:
                        continue
                    hold = holds[v] if p == q else 0
                    c = melodic(p, q, qq, hold)
                    if t >= final_bar:
                        c = (0 if p == q else 2.0) if q is not None else 0
                    if pc not in pcs:
                        c += 4 if strong else (0.5 if q is not None and abs(p - q) <= 2 else 3)
                    if rank[v] == nv - 1 and strong and t % slot_steps == 0 and pc != root:
                        c += 0.8
                    if rank[v] == 0 and t == T - 1 and pc != tonic % 12:
                        c += 3
                    nf = nxt[v][t]
                    if nf is not None and nf[0] - t <= 2 * spb:
                        gap = nf[0] - t
                        dist = abs(p - nf[1])
                        c += 0.4 * max(0, dist - 2 * gap) + (0.5 if gap == 1 and dist > 2 else 0)
                    c += rng.gauss(0, 0.3)
                    cands.append((p, c))
                cands.sort(key=lambda x: x[1])
                options.append(cands[:per_voice] or [(base, 10.0)])
            free = [v for v in range(nv) if t >= entered[v] and fixed[v][t] is None]
            for combo in itertools.product(*options):
                pitches = tuple(x[0] for x in combo)
                c = cost + sum(x[1] for x in combo)
                sounding = [v for v in range(nv) if pitches[v] is not None]
                bass = min((pitches[v] for v in sounding), default=None)
                for i, j in itertools.combinations(sounding, 2):
                    if i not in free and j not in free:
                        continue
                    a, b = pitches[i], pitches[j]
                    ic = abs(a - b) % 12
                    diss = ic not in CONSONANT and not (ic == 5 and min(a, b) != bass)
                    if diss:
                        c += 3 if strong else 0.7
                    if a == b:
                        c += 2
                    hi_v, lo_v = (i, j) if rank[i] < rank[j] else (j, i)
                    if pitches[hi_v] < pitches[lo_v]:
                        c += 3
                    if cur[i] is not None and cur[j] is not None:
                        da, db = a - cur[i], b - cur[j]
                        if da * db > 0 and abs(cur[i] - cur[j]) % 12 == ic and ic in (0, 7):
                            c += 6
                for r in range(nv - 2):
                    u, w = order[r], order[r + 1]
                    if pitches[u] is not None and pitches[w] is not None and pitches[u] - pitches[w] > 12:
                        c += 1
                nh = tuple((holds[v] + 1) if pitches[v] == cur[v] else 0 for v in range(nv))
                new.append((c, pitches, cur, nh, (parent, pitches)))
        new.sort(key=lambda s: s[0])
        states = new[:beam]

    best = states[0]
    grid, node = [], best[4]
    while node is not None:
        node, pitches = node
        grid.append(pitches)
    grid.reverse()

    notes = []
    for p in placed:
        notes += [(p["voice"], on, d, m, "S") for on, d, m in p["notes"]]
    for v in range(nv):
        t = 0
        while t < T:
            if fixed[v][t] is not None or grid[t][v] is None:
                t += 1
                continue
            m, u = grid[t][v], t + 1
            while u < T and fixed[v][u] is None and grid[u][v] == m and (u - t) < 4 * spb:
                u += 1
            notes.append((v, t * step, (u - t) * step, m, "free"))
            t = u
    notes.sort(key=lambda n: (n[1], n[0]))
    return notes, total, [c[0] for c in chords], best[0]


# ---------------------------------------------------------------- output
def write_midi(notes, nv, path, tempo=72):
    from music21 import stream, note, tempo as m21tempo, instrument, meter
    sc = stream.Score()
    for v in range(nv):
        part = stream.Part()
        part.insert(0, instrument.Harpsichord())
        part.insert(0, meter.TimeSignature("4/4"))
        if v == 0:
            part.insert(0, m21tempo.MetronomeMark(number=tempo))
        for vv, on, d, m, _ in notes:
            if vv == v:
                n = note.Note(m)
                n.quarterLength = d
                part.insert(on, n)
        sc.insert(0, part)
    sc.write("midi", fp=str(path))
    return sc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subject", required=True, help="e.g. 'C4/1 G4/1 A4/.5 G4/.5 ...'")
    ap.add_argument("--key", default="C", help="major key tonic, e.g. C, G, F#")
    ap.add_argument("--template", default="wtc1f01")
    ap.add_argument("--plans", default="results/wtc1_events.json")
    ap.add_argument("--name", default="gen01")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beam", type=int, default=40)
    a = ap.parse_args(argv)

    plans = {p["piece"]: p for p in json.loads(Path(a.plans).read_text())}
    template = plans[a.template]
    tonic = 60 + NAMES[a.key[0].upper()] + a.key[1:].count("#") - a.key[1:].count("b")
    subject = parse_subject(a.subject)
    placed = realise(subject, template, tonic)
    notes, total, chords, cost = fill(placed, template, tonic, beam=a.beam, seed=a.seed)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_midi(notes, template["voices"], out / f"{a.name}.mid")

    # analyse the result with the same pipeline used on Bach, for the map
    from fugue.coverage import N, analyse_voices
    from music21 import pitch
    vs = [[N(on, d, pitch.Pitch(midi=m).diatonicNoteNum, m) for vv, on, d, m, _ in notes if vv == v]
          for v in range(template["voices"])]
    res = analyse_voices([v for v in vs if v], a.name)
    res.update({"generated": True, "template": a.template, "key": a.key, "subject_text": a.subject,
                "length": total, "chords": chords, "fill_cost": round(cost, 1),
                "plan": [{k: p[k] for k in ("voice", "onset", "end", "transform", "midi_shift", "clash")}
                         for p in placed],
                "notes": [[v, on, d, m, src] for v, on, d, m, src in notes]})
    gen_path = out / "generated.json"
    gens = json.loads(gen_path.read_text()) if gen_path.exists() else []
    gens = [g for g in gens if g["piece"] != a.name] + [res]
    gen_path.write_text(json.dumps(gens, indent=1))
    print(f"{a.name}: {len(placed)} of {sum(e['material'] == 'SUBJECT' for e in template['events'])} "
          f"template entries placed; {total:.0f} crotchets; {len(notes)} notes; "
          f"subject entries recovered by analyser: {res['full_entries']}; "
          f"coverage {100 * res['cov_all'][0]:.1f}%")
    print(f"wrote {out / (a.name + '.mid')} and {gen_path}")


if __name__ == "__main__":
    main()
