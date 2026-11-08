"""Tokens all the way down: one vocabulary for a piece's frame, its structure and its notes.

A piece is written as
    BOS FORM_* KEY_* METER_* VOICES_n
    BAR POS_p [ENTRY ROLE_* SHIFT_s] VOICE_v PITCH_m DUR_d VOICE_v PITCH_m DUR_d ... POS_p ... BAR ...
    EOS
Time runs on a grid of 12 per crotchet (semiquavers and triplets both land on it). Notes struck together
share one POS token and are written lowest voice first. Pitches are absolute (PITCH_m), except inside a
marked subject statement: there each note after the first is the interval from the one before (INT_k), so a
subject restated at another pitch is the same run of tokens and stating it again is copying. (Intervals
everywhere, tried in the third run, made the model fall into loops of a repeated figure.) The absolute pitch
of every note is also given to the model alongside the tokens (aux_of). ENTRY marks where a fugue's subject (or answer, or
inversion) begins, in which voice and at what transposition from the first statement; the notes that follow
are written like any others, so stating the subject is something the model has to do, not something placed."""
import bisect

TPB = 12                                                  # grid steps per crotchet
MAX_VOICES = 8
LO, HI = 21, 108                                          # piano range
DURS = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 30, 36, 42, 48, 60, 72, 84, 96, 120, 144, 192]
FORMS = ["FUGUE", "INVENTION", "CHORALE", "OTHER"]
METERS = ["2/2", "3/2", "4/2", "2/4", "3/4", "4/4", "5/4", "6/4", "3/8", "6/8", "9/8", "12/8", "6/16", "12/16", "24/16", "OTHER"]
ROLES = ["x1", "inv", "aug", "dim", "other"]
MAX_INT = 24                                              # intervals within a voice, up to two octaves
MAX_POS = 8 * TPB * 2                                     # positions in a bar, up to 16 crotchets

def _build():
    v = ["PAD", "BOS", "EOS", "BAR", "ENTRY"]
    v += [f"FORM_{f}" for f in FORMS]
    v += [f"KEY_{t}_{m}" for m in ("major", "minor") for t in range(12)]
    v += [f"METER_{m}" for m in METERS]
    v += [f"VOICES_{n}" for n in range(1, MAX_VOICES + 1)]
    v += [f"ROLE_{r}" for r in ROLES]
    v += [f"SHIFT_{s}" for s in range(12)]
    v += [f"POS_{p}" for p in range(MAX_POS)]
    v += [f"VOICE_{i}" for i in range(MAX_VOICES)]
    v += [f"PITCH_{m}" for m in range(LO, HI + 1)]
    v += [f"DUR_{d}" for d in DURS]
    v += ["CONT"]          # training only: this window begins partway through the piece (see data.py)
    v += [f"INT_{i}" for i in range(-MAX_INT, MAX_INT + 1)]
    return v

VOCAB = _build()
ID = {t: i for i, t in enumerate(VOCAB)}

def dur_token(steps):
    i = bisect.bisect_left(DURS, steps)
    if i == len(DURS): return ID[f"DUR_{DURS[-1]}"]
    if i > 0 and steps - DURS[i - 1] < DURS[i] - steps: i -= 1
    return ID[f"DUR_{DURS[i]}"]

def meter_name(num, den):
    return f"{num}/{den}" if f"{num}/{den}" in METERS else "OTHER"

def bars(meters, end):
    """[(step, meter name)] for every bar line, the meter named where it changes. meters: [[step, num, den]]."""
    out, ms = [], sorted(meters) or [[0, 4, 4]]
    for i, (a, num, den) in enumerate(ms):
        b = ms[i + 1][0] if i + 1 < len(ms) else end
        L = max(1, round(num * 4 / den * TPB)); t = a
        while t < b: out.append((t, meter_name(num, den) if t == a else None)); t += L
    return out

def encode(piece, shift=0, relative="entries"):
    """piece: {"form", "key": [tonic_pc, "major"|"minor"], "meters", "bar_offset" (steps), "voices": [[[on, d, m], ...]]
    top first, "entries": [{"voice", "onset", "role", "shift"}]}. Returns token ids, transposed by `shift`
    semitones (None if that takes a note out of range)."""
    voices = piece["voices"][:MAX_VOICES]; nv = len(voices)
    num, den = (piece["meters"][0][1], piece["meters"][0][2]) if piece["meters"] else (4, 4)
    meter = f"{num}/{den}" if f"{num}/{den}" in METERS else "OTHER"
    tonic, mode = piece["key"]
    out = [ID["BOS"], ID[f"FORM_{piece['form']}"], ID[f"KEY_{(tonic + shift) % 12}_{mode}"], ID[f"METER_{meter}"], ID[f"VOICES_{nv}"]]
    # a pickup is written as the end of a full first bar, so every bar is whole and bar lines need no lengths
    L0 = max(1, round(num * 4 / den * TPB)); lead = (L0 - piece.get("bar_offset", 0)) % L0
    events = {}                                         # step -> [(kind, voice, ...)], kind 0 = entry, 1 = note
    spans = {}
    for e in piece.get("entries", []):
        if "end" in e: spans.setdefault(e["voice"], []).append((e["onset"] - 1e-6, e["end"] - 1e-6))
    for v, notes in enumerate(voices):
        for on, d, m in notes:
            m += shift
            if m < LO or m > HI: return None
            inside = any(a <= on < b for a, b in spans.get(v, []))
            t = round(on * TPB) + lead; events.setdefault(t, []).append((1, v, m, max(1, round(d * TPB)), inside))
    for e in piece.get("entries", []):
        if e["voice"] < nv: events.setdefault(round(e["onset"] * TPB) + lead, []).append((0, e["voice"], e["role"], e["shift"]))
    end = max(events) + 1
    meters = [[0, num, den]] + [[round(o * TPB) + lead, n_, d_] for o, n_, d_ in piece["meters"][1:]]
    bl = bars(meters, end); bi = 0; cur = 0; flags = []
    for t in sorted(events):
        while bi < len(bl) and bl[bi][0] <= t:
            cur = bl[bi][0]; out.append(ID["BAR"])
            if bl[bi][1] and bi > 0: out.append(ID[f"METER_{bl[bi][1]}"])
            bi += 1
        out.append(ID[f"POS_{min(MAX_POS - 1, t - cur)}"])
        # lowest voice first (highest index); an entry marker just before its voice's first note
        for ev in sorted(events[t], key=lambda x: (-x[1], x[0], x[2] if x[0] else 0)):
            if ev[0] == 0:
                out += [ID["ENTRY"], ID[f"ROLE_{ev[2] if ev[2] in ROLES else 'other'}"], ID[f"SHIFT_{ev[3] % 12}"], ID[f"VOICE_{ev[1]}"]]
            else:
                out += [ID[f"VOICE_{ev[1]}"], ID[f"PITCH_{ev[2]}"], dur_token(ev[3])]
                flags.append(ev[4])
    out.append(ID["EOS"])
    if relative == "all": return to_relative(out)
    if relative == "entries": return to_relative(out, flags)
    return out

def to_relative(ids, flags=None):
    """PITCH tokens to intervals from the voice's previous note: all of them (flags None), or those whose flag
    is set (notes inside a subject statement, in order of appearance), except the first note of each statement
    and leaps beyond MAX_INT."""
    last, prev_in, out, i, k = {}, {}, [], 0, 0
    while i < len(ids):
        t = VOCAB[ids[i]]
        if t.startswith("VOICE_") and i + 1 < len(ids) and VOCAB[ids[i + 1]].startswith("PITCH_"):
            v, m = int(t[6:]), int(VOCAB[ids[i + 1]][6:])
            inside = True if flags is None else flags[k]
            rel = inside and v in last and abs(m - last[v]) <= MAX_INT and (flags is None or prev_in.get(v, False))
            out += [ids[i], ID[f"INT_{m - last[v]}"] if rel else ids[i + 1]]
            last[v] = m; prev_in[v] = inside; i += 2; k += 1; continue
        out.append(ids[i]); i += 1
    return out

def aux_of(ids):
    """For each token, the absolute pitch it puts down (PITCH or INT tokens), else 0: side information that
    lets the model see the harmony while it writes intervals."""
    last, cur, aux = {}, None, []
    for i in ids:
        t = VOCAB[i]
        if t.startswith("VOICE_"): cur = int(t[6:]); aux.append(0)
        elif t.startswith("PITCH_"): last[cur] = int(t[6:]); aux.append(last[cur])
        elif t.startswith("INT_"):
            last[cur] = min(HI, max(LO, last.get(cur, 60) + int(t[4:]))); aux.append(last[cur])
        else: aux.append(0)
    return aux

def decode(ids):
    """Token ids back to {"header": {...}, "notes": [{v, on, d, m}], "entries": [{v, on, role, shift}]}; times in crotchets."""
    toks = [VOCAB[i] for i in ids]; head = {}; notes = []; entries = []
    L, start, t, first, i = 4 * TPB, 0, 0, True, 0
    last_pitch = {}
    def length(name):
        if name == "OTHER": return 4 * TPB
        n, d = name.split("/"); return round(int(n) * 4 / int(d) * TPB)
    while i < len(toks):
        k = toks[i]
        if k.startswith(("FORM_", "KEY_", "VOICES_")): head[k.split("_")[0]] = k[k.index("_") + 1:]
        elif k.startswith("METER_"):
            if "METER" not in head: head["METER"] = k[6:]
            L = length(k[6:])
        elif k == "BAR":
            if first: first = False
            else: start += L_prev
            L_prev = L
        elif k.startswith("POS_"): t = start + int(k[4:])
        elif k == "ENTRY" and i + 3 < len(toks):
            entries.append({"v": int(toks[i + 3][6:]) if toks[i + 3].startswith("VOICE_") else 0, "on": t / TPB,
                            "role": toks[i + 1][5:], "shift": int(toks[i + 2][6:]) if toks[i + 2].startswith("SHIFT_") else 0})
            i += 3
        elif k.startswith("VOICE_") and i + 2 < len(toks) and toks[i + 1].startswith(("PITCH_", "INT_")) and toks[i + 2].startswith("DUR_"):
            v = int(k[6:]); p = toks[i + 1]
            m = int(p[6:]) if p.startswith("PITCH_") else min(HI, max(LO, last_pitch.get(v, 60) + int(p[4:])))
            last_pitch[v] = m
            notes.append({"v": v, "on": t / TPB, "d": int(toks[i + 2][4:]) / TPB, "m": m}); i += 2
        elif k == "EOS": break
        i += 1
    return {"header": head, "notes": notes, "entries": entries}
