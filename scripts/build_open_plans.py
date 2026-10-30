"""Entry plans from the open edition (both books): results/plans_open.json.

Same fields as results/plans.json, but every fugue of both books, and computed from
data/open_wtc.json instead of the Huron encoding. Diatonic note numbers are inferred from
MIDI pitch and the key (chromatic notes take the scale degree below)."""
import json, sys
sys.path.insert(0, "src")
from fugue.coverage import N, analyse_voices, matches

MAJ = [0, 2, 4, 5, 7, 9, 11]; MIN = [0, 2, 3, 5, 7, 8, 10]
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

def dia(m, tonic, sc):
    rel = m - tonic; o, pc = rel // 12, rel % 12
    return max(i for i in range(7) if sc[i] <= pc) + 7 * o + 35

def tolerant(v, pat, need=0.85):
    """Statements of pat in voice v allowing a few altered intervals (tonal answers, cadential changes).
    Yields (start, transform). Rhythm must agree for most notes (the last note may differ);
    transforms: x1, INV, AUG (x2), DIM (x0.5)."""
    L = len(pat); pi = [pat[j + 1].dia - pat[j].dia for j in range(L - 1)]; pd = [n.dur for n in pat]
    for i in range(len(v) - L + 1):
        w = v[i:i + L]
        if any(abs((w[j + 1].on - w[j].on) - w[j].dur) > 1e-6 for j in range(L - 1)) and False: pass
        vi = [w[j + 1].dia - w[j].dia for j in range(L - 1)]
        for sc, tag in ((1, "x1"), (2, "AUG"), (0.5, "DIM")):
            rh = sum(abs(w[j].dur - pd[j] * sc) < 1e-6 for j in range(L - 1)) / max(1, L - 1)
            if rh < need: continue
            for inv in (False, True):
                ok = sum((vi[j] == (-pi[j] if inv else pi[j])) or (j == 0 and abs(vi[j] - (-pi[j] if inv else pi[j])) <= 1) for j in range(L - 1)) / max(1, L - 1)
                if ok >= need: yield i, ("INV+" if inv else "") + tag; break
            else: continue
            break

fugues = json.load(open("data/open_wtc.json"))["fugues"]
plans = []
for f in fugues:
    sc = MIN if f["mode"] == "minor" else MAJ
    vs = [[N(o, d, dia(m, f["tonic_pc"], sc), m) for o, d, m in v] for v in f["voices"]]
    r = analyse_voices(vs, f["piece"])
    order = sorted(range(len(vs)), key=lambda k: vs[k][0].on)
    vs = [vs[k] for k in order]                                       # voices in order of entry, as the analyser has them
    # the subject: the longest opening of the first voice that recurs in full in the other voices
    e2 = vs[1][0].on; head = [n for n in vs[0] if n.on < e2 - 1e-9]
    pat = head
    def statements(pat):
        out = []
        for vi, v in enumerate(vs):
            last_end = -1
            for i, tr in tolerant(v, pat):
                if v[i].on < last_end - 1e-9: continue                       # no overlapping statements in one voice
                k = len(pat); last_end = v[i + k - 1].on + v[i + k - 1].dur
                out.append({"voice": vi, "onset": v[i].on, "end": last_end, "transform": tr,
                            "dia_shift": v[i].dia - pat[0].dia, "midi_shift": v[i].midi - pat[0].midi})
        return out
    for L in range(len(head), 4, -1):
        if len(statements(head[:L])) - 1 >= len(vs) - 1: pat = head[:L]; break
    ents = statements(pat)
    ents.sort(key=lambda e: e["onset"])
    subj = [[n.on - pat[0].on, n.dur, n.dia, n.midi] for n in pat]
    overlaps = sum(1 for a in ents for b in ents if a is not b and a["voice"] != b["voice"] and a["onset"] < b["onset"] < a["end"])
    plans.append({
        "id": f["piece"], "book": f["book"], "number": f["number"], "key": NAMES[f["tonic_pc"]], "mode": f["mode"], "time": f["time"],
        "voices": len(vs), "voice_ranges": [[min(n.midi for n in v), max(n.midi for n in v)] for v in vs], "length": r["length"],
        "subject_span": subj[-1][0] + subj[-1][1], "subject_first_midi": subj[0][3], "subject_notes": len(subj),
        "entries": ents,
        "n_entries": len(ents), "n_inverted": sum(e["transform"].startswith("INV") for e in ents), "n_overlaps": overlaps,
        "cov_subject": round(r["cov_subject"][0], 3), "cov_all": round(r["cov_all"][0], 3),
    })
json.dump(plans, open("results/plans_open.json", "w"), separators=(",", ":"))
import statistics as st
print(len(plans), "plans; entries median", st.median(p["n_entries"] for p in plans), "; subject coverage median %.1f%%, all material %.1f%%" % (100 * st.median(p["cov_subject"] for p in plans), 100 * st.median(p["cov_all"] for p in plans)))
for p in plans:
    if p["n_entries"] < 5 or p["n_entries"] > 35: print("  unusual:", p["id"], p["n_entries"], "entries, subject", p["subject_notes"], "notes")
