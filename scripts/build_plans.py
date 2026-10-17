"""Compact entry plans for the in-browser generator: results/plans.json.

Per WTC fugue: key and mode (read from the **kern key token), voices, voice ranges,
the subject's span, and its subject entries (voice, onset, end, transform, shifts).
"""
import json
import re
import sys
from pathlib import Path

events = json.loads(Path(sys.argv[1]).read_text())
kern_dir = Path(sys.argv[2])
plans = []
for f in events:
    text = (kern_dir / f"{f['piece']}.krn").read_text(errors="replace")
    m = re.search(r"^\*([A-Ga-g])([#-]*):", text, re.M)
    letter, acc = m.group(1), m.group(2)
    mode = "major" if letter.isupper() else "minor"
    key = letter.upper() + acc.replace("-", "b")
    subj = f["subject"]
    ents = sorted((e for e in f["events"] if e["material"] == "SUBJECT"), key=lambda e: e["onset"])
    overlaps = sum(1 for a in ents for b in ents
                   if a is not b and a["voice"] != b["voice"] and a["onset"] < b["onset"] < a["end"])
    plans.append({
        "id": f["piece"], "number": int(f["piece"][-2:]), "key": key, "mode": mode,
        "voices": f["voices"], "voice_ranges": f["voice_ranges"],
        "subject_span": subj[-1][0] + subj[-1][1],
        "subject_first_midi": subj[0][3],
        "entries": [{k: e[k] for k in ("voice", "onset", "end", "transform", "dia_shift", "midi_shift")}
                    for e in ents],
        "n_entries": len(ents), "n_inverted": sum(e["transform"].startswith("INV") for e in ents),
        "n_overlaps": overlaps,
        # usable as a generation plan: enough entries found, and not No. 4 (four-note
        # subject, triple fugue: many spurious matches)
        "usable": 7 <= len(ents) <= 30 and f["piece"] != "wtc1f04",
    })
Path(sys.argv[3]).write_text(json.dumps(plans, separators=(",", ":")))
for p in plans:
    print(f"No.{p['number']:2} {p['key']:3} {p['mode']:5} v{p['voices']} entries {p['n_entries']:2} inv {p['n_inverted']:2} overlaps {p['n_overlaps']:2}")
