"""Build data/open_wtc.json from Kyle Rother's open edition of the 48 fugues (CC BY 4.0).

Each .ly source writes every voice on its own staff; LilyPond renders one MIDI track per staff,
so each track is one voice. Chords (a few final cadences) are reduced to one note: the lowest
for the bottom voice, the highest otherwise.

usage: python scripts/open_wtc.py <midi_dir> <ly_dir_root> out.json
"""
import glob, json, os, re, sys
import mido

# Known slips in the edition, found where a voice suddenly crosses above its neighbour and stays there:
# piece -> [(staff index among staves with notes, from onset in crotchets, semitone shift)]
FIXES = {
    "wtc1f04": [(2, 96.0, -12)],   # mezzo written an octave high from bar 25 (visible in the edition's PDF: it rises above the alto)
}
MIN_NOTES = 20                     # a "staff" with fewer notes is extra notes of a final chord, not a voice

KEYS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

def ly_key(path):
    s = open(path, encoding="utf-8").read()
    m = re.search(r"\\key\s+([a-g])(is|es|s)?\s*\\(major|minor)", s)
    pc = KEYS[m.group(1)] + {"is": 1, "es": -1, "s": -1, None: 0}[m.group(2)]
    t = re.search(r"\\time\s+(\d+)/(\d+)", s)
    return pc % 12, m.group(3), (int(t.group(1)), int(t.group(2))) if t else (4, 4)

def voices(midi_path, fixes=()):
    mf = mido.MidiFile(midi_path); tpq = mf.ticks_per_beat
    out = []
    for tr in mf.tracks:
        t, on, notes = 0, {}, []
        for msg in tr:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0: on.setdefault(msg.note, t)
            elif msg.type in ("note_off", "note_on") and msg.note in on:
                s = on.pop(msg.note); notes.append([s / tpq, (t - s) / tpq, msg.note])
        if notes: out.append(sorted(notes))
    for staff, start, shift in fixes:
        for n in out[staff]:
            if n[0] >= start: n[2] += shift
    out = [v for v in out if len(v) >= MIN_NOTES]
    out.sort(key=lambda v: -sum(n[2] for n in v) / len(v))           # top voice first
    clean = []
    for k, v in enumerate(out):
        bottom = k == len(out) - 1; byon = {}
        for n in v: byon.setdefault(round(n[0], 4), []).append(n)
        mono = [(min if bottom else max)(g, key=lambda n: n[2]) for _, g in sorted(byon.items())]
        for a, b in zip(mono, mono[1:]): a[1] = min(a[1], b[0] - a[0])  # no overlaps inside a voice
        clean.append([[round(x, 4) for x in n[:2]] + [n[2]] for n in mono if n[1] > 1e-6])
    return clean

def main(midi_dir, ly_root, out):
    data = []
    for mp in sorted(glob.glob(os.path.join(midi_dir, "wtc*f*.mid*"))):
        name = os.path.basename(mp).split(".")[0]; book, num = int(name[3]), int(name[5:7])
        ly = [p for p in glob.glob(f"{ly_root}/b{book}/book{book}/fugue{num}_*/fugue{num}_*.ly") if "prelude" not in os.path.basename(p)][0]
        pc, mode, ts = ly_key(ly)
        data.append({"piece": name, "book": book, "number": num, "tonic_pc": pc, "mode": mode, "time": ts, "voices": voices(mp, FIXES.get(name, ()))})
    json.dump({"source": "Kyle Rother, open score edition of the 48 fugues of the Well-Tempered Clavier (2015), OpenUCT, CC BY 4.0",
               "fugues": data}, open(out, "w"))
    print(len(data), "fugues,", sum(len(f["voices"]) for f in data), "voices,", sum(len(v) for f in data for v in f["voices"]), "notes")

if __name__ == "__main__":
    main(*sys.argv[1:4])
