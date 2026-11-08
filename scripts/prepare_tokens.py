"""Gather the open corpora into one list of pieces for the token model (data/tx_pieces.json).

    python scripts/prepare_tokens.py

Sources: data/open_wtc.json (Rother's WTC, CC BY 4.0) with fugue entries from results/plans_open.json, and
data/open_bach.json (Mutopia, Public Domain / CC BY / CC BY-SA). Each piece gets a form, a key (the WTC's is
given; Mutopia's is estimated: the final bass note is taken as tonic, the mode from Krumhansl-Kessler
profiles), a meter and where its first full bar begins (taken from the last onset, since Bach ends on a
downbeat). The six fugues held out from the counterpoint model stay held out here, with a few percent of
the Mutopia pieces."""
import json, re, zlib, os, sys
sys.path.insert(0, "src")
from fugue.subject_find import find_entries

HELD = {"wtc1f02", "wtc1f11", "wtc1f20", "wtc2f05", "wtc2f14", "wtc2f23"}
TPB = 12
KK_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KK_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

def corr(a, b):
    ma, mb = sum(a) / 12, sum(b) / 12
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / ((sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5 + 1e-9)

def estimate_key(voices):
    h = [0.0] * 12
    for v in voices:
        for _, d, m in v: h[m % 12] += d
    last = max(n[0] for v in voices for n in v)
    tonic = min((n for v in voices for n in v if n[0] >= last - 1e-6), key=lambda n: n[2])[2] % 12
    rot = h[tonic:] + h[:tonic]
    return [tonic, "minor" if corr(rot, KK_MIN) > corr(rot, KK_MAJ) else "major"]

def bar_offset(voices, meters):
    num, den = (meters[0][1], meters[0][2]) if meters else (4, 4)
    L = round(num * 4 / den * TPB)
    last = max(round(n[0] * TPB) for v in voices for n in v)
    if len(meters) > 1: return 0                       # meter changes: trust the file's bar lines
    return last % L

def merge_voices(tracks, cap=8):
    """LilyPond writes a track for every voice, including the short-lived ones a staff splits into for a few
    bars. Fragments are joined to the longer voice nearest in pitch that is silent while they sound; whatever
    is left beyond `cap` is joined to its nearest neighbour (as chords)."""
    info = sorted(([n for n in t], sum(n[2] for n in t) / len(t)) for t in tracks if t)
    info.sort(key=lambda x: -sum(n[1] for n in x[0]))
    slots = []                                         # [notes, mean pitch, busy intervals]
    def overlap(busy, notes):
        return sum(max(0, min(e, o + d) - max(b, o)) for o, d, _ in notes for b, e in busy if b < o + d and o < e)
    for notes, mean in info:
        span = sum(n[1] for n in notes)
        best = None
        for sl in slots:
            if abs(sl[1] - mean) <= 12 and overlap(sl[2], notes) <= 0.05 * span:
                if best is None or abs(sl[1] - mean) < abs(best[1] - mean): best = sl
        if best is None: slots.append([list(notes), mean, [(o, o + d) for o, d, _ in notes]])
        else:
            best[0] += notes; best[2] += [(o, o + d) for o, d, _ in notes]
            best[1] = sum(n[2] for n in best[0]) / len(best[0])
    slots.sort(key=lambda x: -x[1])
    while len(slots) > cap:                            # join the smallest voice to its neighbour in pitch
        i = min(range(len(slots)), key=lambda k: len(slots[k][0]))
        j = min((k for k in range(len(slots)) if k != i), key=lambda k: abs(slots[k][1] - slots[i][1]))
        slots[j][0] += slots[i][0]; slots.pop(i)
    return [sorted(sl[0]) for sl in slots]

def form_of(title, bwv):
    t = title.lower() + " " + bwv.lower(); n = int(re.match(r"BWV(\d+)", bwv).group(1)) if re.match(r"BWV(\d+)", bwv) else -1
    if "fug" in t or "contrapunctus" in t or "canon" in t: return "FUGUE"
    if "invention" in t or "sinfonia" in t or 772 <= n <= 801: return "INVENTION"
    if "choral" in t or 250 <= n <= 438: return "CHORALE"
    return "OTHER"

ROLE = {"x1": "x1", "INV+x1": "inv", "AUG": "aug", "DIM": "dim"}

def role(transform):
    return ROLE.get(transform, "other")

def main():
    pieces = []
    wtc = json.load(open("data/open_wtc.json"))["fugues"]
    plans = {p["id"]: p for p in json.load(open("results/plans_open.json"))}
    for f in wtc:
        num, den = json.loads(f["time"]) if isinstance(f["time"], str) else f["time"]
        meters = [[0, num, den]]
        # plans number voices in order of entry; the data numbers them top first
        order = sorted(range(len(f["voices"])), key=lambda i: f["voices"][i][0][0])
        entries = [{"voice": order[e["voice"]], "onset": e["onset"], "role": role(e["transform"]), "shift": e["midi_shift"]}
                   for e in plans[f["piece"]]["entries"]] if f["piece"] in plans else []
        pieces.append({"id": f["piece"], "source": "rother-wtc", "licence": "CC BY 4.0", "form": "FUGUE", "split": "val" if f["piece"] in HELD else "train",
                       "key": [f["tonic_pc"], f["mode"]], "meters": meters, "bar_offset": bar_offset(f["voices"], meters),
                       "voices": f["voices"], "entries": entries})
    if os.path.exists("data/open_bach.json"):
        seen = set()
        for p in json.load(open("data/open_bach.json"))["pieces"]:
            # the same music often comes out of several files (paper sizes, part and score versions)
            sig = hash(tuple(sorted((round(o, 2), m) for v in p["voices"] for o, _, m in v)))
            if sig in seen: continue
            seen.add(sig)
            if len(p["voices"]) > 12: continue             # orchestral and choral scores: doublings, not counterpoint
            v = merge_voices(p["voices"])
            meters = [[m[0], m[1], m[2]] for m in p["meters"]] or [[0, 4, 4]]
            form, key = form_of(p["title"], p["id"]), estimate_key(v)
            entries = []
            if form in ("FUGUE", "INVENTION"):                # the same subject finder as the WTC plans
                ents, _ = find_entries(v, key[0], key[1])
                entries = [{"voice": e["voice"], "onset": e["onset"], "role": role(e["transform"]), "shift": e["midi_shift"]} for e in ents]
            # a fugue whose subject the finder could not locate carries no entry markers; calling it a fugue would
            # teach the model that fugues may come without them, so it is filed with everything else
            if form in ("FUGUE", "INVENTION") and not entries: form = "OTHER"
            pieces.append({"id": p["id"], "source": "mutopia", "licence": p["licence"], "form": form,
                           "split": "val" if zlib.crc32(p["bwv"].encode()) % 33 == 0 else "train",
                           "key": key, "meters": meters, "bar_offset": bar_offset(v, meters), "voices": v, "entries": entries})
    json.dump({"pieces": pieces}, open("data/tx_pieces.json", "w"), separators=(",", ":"))
    from collections import Counter
    print(len(pieces), "pieces", Counter((p["source"], p["form"], p["split"]) for p in pieces))
    print("pieces with entries marked:", Counter((p["source"], p["form"]) for p in pieces if p["entries"]))

if __name__ == "__main__":
    main()
