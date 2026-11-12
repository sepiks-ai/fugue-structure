"""Entry plans from the open edition (both books): results/plans_open.json.

Same fields as results/plans.json, but every fugue of both books, and computed from
data/open_wtc.json instead of the Huron encoding. Diatonic note numbers are inferred from
MIDI pitch and the key (chromatic notes take the scale degree below)."""
import json, sys
sys.path.insert(0, "src")
from fugue.coverage import N, analyse_voices, matches

NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

from fugue.subject_find import dia, tolerant, MAJ, MIN

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
