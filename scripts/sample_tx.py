"""Write a piece with the token composer.

    py scripts/sample_tx.py [--ckpt=results/tx/best.pt] [--form=FUGUE] [--key=2_minor] [--meter=4/4] [--voices=3]
                            [--temperature=0.95] [--top_p=0.95] [--max_tokens=6000] [--seed=0] [--out=results/tx/sample]

Only the header is given; everything after it (bars, entries, notes, the end) is the model's. A grammar keeps
the tokens well formed (a voice is followed by a pitch, a pitch by a length; positions only move forward
within a bar; at most --voices voices) and nothing else. Writes <out>.mid and <out>.json."""
import os, sys, json, random, torch, mido
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fugue.tx import tokens as T
from fugue.tx.model import Composer

def args():
    a = dict(ckpt="results/tx/best.pt", form="FUGUE", key="2_minor", meter="4/4", voices=3, temperature=0.95, top_p=0.95,
             max_tokens=6000, seed=0, out="results/tx/sample", cpu=0)
    for x in sys.argv[1:]:
        k, v = x[2:].split("=", 1) if "=" in x else (x[2:], "1"); a[k] = type(a[k])(v)
    return a

GROUP = {}
for i, t in enumerate(T.VOCAB): GROUP.setdefault(t.split("_")[0] if "_" in t else t, []).append(i)

class Grammar:
    def __init__(s, nv, meter):
        s.nv, s.prev, s.pos, s.voice = nv, "BAR", -1, None
        s.last = {}                                       # voice -> last pitch, for interval tokens
        s.L = s.length(meter)
    @staticmethod
    def length(m):
        if m == "OTHER": return 4 * T.TPB
        n, d = m.split("/"); return round(int(n) * 4 / int(d) * T.TPB)
    def allowed(s):
        p, ok = s.prev, set()
        if p == "BAR": ok |= set(GROUP["METER"]) | {T.ID[f"POS_{i}"] for i in range(min(s.L, T.MAX_POS))}
        elif p == "METER": ok |= {T.ID[f"POS_{i}"] for i in range(min(s.L, T.MAX_POS))}
        elif p in ("POS", "DUR"):
            ok |= {T.ID[f"VOICE_{v}"] for v in range(s.nv) if s.voice is None or v <= s.voice} | {T.ID["ENTRY"]}
            if p == "DUR": ok |= {T.ID[f"POS_{i}"] for i in range(s.pos + 1, min(s.L, T.MAX_POS))} | {T.ID["BAR"], T.ID["EOS"]}
        elif p == "ENTRY": ok |= set(GROUP["ROLE"])
        elif p == "ROLE": ok |= set(GROUP["SHIFT"])
        elif p == "SHIFT": ok |= {T.ID[f"VOICE_{v}"] for v in range(s.nv)}
        elif p == "EVOICE": ok |= {T.ID[f"VOICE_{s.voice}"]}
        elif p == "VOICE":
            if s.voice not in s.last: ok |= set(GROUP["PITCH"])        # a voice's first note is absolute
            else:
                m0 = s.last[s.voice]
                ok |= {T.ID[f"INT_{k}"] for k in range(-T.MAX_INT, T.MAX_INT + 1) if T.LO <= m0 + k <= T.HI}
                ok |= set(GROUP["PITCH"])
        elif p in ("PITCH", "INT"): ok |= set(GROUP["DUR"])
        return sorted(ok)
    def step(s, i):
        t = T.VOCAB[i]; g = t.split("_")[0] if "_" in t else t
        if g == "BAR": s.pos, s.voice = -1, None
        elif g == "METER": s.L = s.length(t[6:])
        elif g == "POS": s.pos, s.voice = int(t[4:]), None
        elif g == "VOICE":
            s.voice = int(t[6:])
            if s.prev == "SHIFT": g = "EVOICE"
        elif g == "PITCH": s.last[s.voice] = int(t[6:])
        elif g == "INT": s.last[s.voice] = s.last[s.voice] + int(t[4:])
        s.prev = g
        return s.last[s.voice] if g in ("PITCH", "INT") else 0

def recurrence(notes, bar=4.0, k=4):
    """How much the music repeats itself: for every pair of bars at least eight bars apart, the share of
    four-note figures (intervals and lengths, at any pitch) they have in common; averaged. Bach's fugues:
    0.02 to 0.05 (WTC I/2, I/11, II/5). The first two runs of this model: 0.002."""
    vs = {}
    for n in notes:
        d = vs.setdefault(n["v"], {})
        if n["on"] not in d or n["m"] > d[n["on"]]["m"]: d[n["on"]] = n          # top note of a chord
    if not vs: return None
    end = max(n["on"] + n["d"] for n in notes); B = int(end // bar) + 1; G = [set() for _ in range(B)]
    for line in vs.values():
        line = sorted(line.values(), key=lambda n: n["on"])
        for i in range(len(line) - k):
            seg = line[i:i + k + 1]
            G[int(seg[0]["on"] // bar)].add(tuple((seg[j + 1]["m"] - seg[j]["m"], round(seg[j]["d"] * 4)) for j in range(k)))
    sims = [len(G[i] & G[j]) / len(G[i] | G[j]) for i in range(B) for j in range(i + 8, B) if G[i] and G[j]]
    return round(sum(sims) / len(sims), 4) if sims else None

def entry_fidelity(piece, k=8):
    """How faithfully the model states its own subject: for each entry after the first, the share of the first k
    melodic intervals in that voice that match the first entry's (negated for an inversion). Its own measure,
    from the model's own markers; the grammar does not enforce it."""
    ents = sorted(piece["entries"], key=lambda e: e["on"])
    def line(e):
        return [n["m"] for n in sorted((n for n in piece["notes"] if n["v"] == e["v"] and n["on"] >= e["on"] - 1e-9), key=lambda n: n["on"])][:k + 1]
    if len(ents) < 2: return None
    ref = line(ents[0]); ri = [b - a for a, b in zip(ref, ref[1:])]
    scores = []
    for e in ents[1:]:
        l = line(e); li = [b - a for a, b in zip(l, l[1:])]
        sign = -1 if e["role"] == "inv" else 1
        n = min(len(ri), len(li))
        if n: scores.append(sum(li[j] == sign * ri[j] for j in range(n)) / n)
    return round(sum(scores) / len(scores), 3) if scores else None

@torch.no_grad()
def main():
    a = args(); torch.manual_seed(a["seed"]); random.seed(a["seed"])
    dev = "cuda" if torch.cuda.is_available() and not a["cpu"] else "cpu"
    ck = torch.load(a["ckpt"], map_location=dev)
    ck["cfg"].setdefault("aux", False)
    model = Composer(**ck["cfg"]).to(dev).eval(); model.load_state_dict(ck["model"])
    ctx = ck.get("args", {}).get("ctx", 2048) if "args" in ck else 2048
    head = [T.ID["BOS"], T.ID[f"FORM_{a['form']}"], T.ID[f"KEY_{a['key']}"], T.ID[f"METER_{a['meter']}"], T.ID[f"VOICES_{a['voices']}"]]
    seq, g = list(head), Grammar(a["voices"], a["meter"])
    nxt = T.ID["BAR"]; seq.append(nxt); aux = [0] * len(seq); g.step(nxt)
    while len(seq) < a["max_tokens"]:
        window = seq if len(seq) <= ctx else head + seq[-(ctx - len(head)):]
        wa = aux if len(aux) <= ctx else aux[:len(head)] + aux[-(ctx - len(head)):]
        logits = model(torch.tensor([window], device=dev), torch.tensor([wa], device=dev) if ck["cfg"].get("aux") else None)[0, -1].float()
        mask = torch.full_like(logits, float("-inf")); ok = g.allowed(); mask[ok] = 0
        p = torch.softmax((logits + mask) / a["temperature"], -1)
        sp, si = p.sort(descending=True); keep = sp.cumsum(0) - sp < a["top_p"]
        sp = sp * keep; i = int(si[torch.multinomial(sp / sp.sum(), 1)])
        seq.append(i); aux.append(g.step(i))
        if i == T.ID["EOS"]: break
        if len(seq) % 500 == 0: print(len(seq), "tokens", flush=True)
    piece = T.decode(seq)
    piece["entry_fidelity"] = entry_fidelity(piece)
    piece["recurrence"] = recurrence(piece["notes"])
    os.makedirs(os.path.dirname(a["out"]) or ".", exist_ok=True)
    json.dump({"args": a, "tokens": len(seq), "ended": seq[-1] == T.ID["EOS"], **piece}, open(a["out"] + ".json", "w"))
    mf = mido.MidiFile(ticks_per_beat=480); tr = mido.MidiTrack(); mf.tracks.append(tr); ev = []
    for n in piece["notes"]:
        ev += [(round(n["on"] * 480), 1, n["m"], n["v"]), (round((n["on"] + n["d"]) * 480), 0, n["m"], n["v"])]
    t = 0
    for at, on, m, v in sorted(ev):
        tr.append(mido.Message("note_on" if on else "note_off", note=m, velocity=80 if on else 0, channel=v % 16, time=at - t)); t = at
    mf.save(a["out"] + ".mid")
    print(f"{len(piece['notes'])} notes, {len(piece['entries'])} entries (subject stated with {piece['entry_fidelity']} of its intervals), "
          f"recurrence {piece['recurrence']} (Bach 0.02-0.05), {len(seq)} tokens, ended: {seq[-1] == T.ID['EOS']}")

if __name__ == "__main__":
    main()
