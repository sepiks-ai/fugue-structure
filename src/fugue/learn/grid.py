"""The semiquaver grid and the categorical features the counterpoint model sees.

Each voice, at each step, either starts a note, holds the note it has, or rests. The model
predicts the next step of one voice from:
  - its own last 12 steps, as intervals from its current pitch p0
  - for up to two voices above and two below, 8 steps around now (-4..+4), also as intervals from p0;
    any of these may be "unknown" (not yet written), which is how the model is used when generating
  - p0's scale degree and register relative to the key, the step's position in the bar and its
    metrical weight, and the voice's role (which voice of how many)
All features are small categorical values; the network sums one learned vector per feature.
The same code is mirrored in web/learned.js, which must stay in step with it.
"""
import numpy as np

STEP = 0.25                         # crotchets per grid step
OWN = 12                            # own history steps
OFFS = (-4, -3, -2, -1, 0, 1, 2, 4) # other-voice offsets
SLOTS = ("above1", "above2", "below1", "below2")
IV = 16                             # target and own-history intervals clamp to +-16
OV = 30                             # other-voice intervals clamp to +-30
N_OUT = 2 * IV + 1 + 2              # 33 onset intervals, HOLD, REST
HOLD, REST = 2 * IV + 1, 2 * IV + 2

# feature tables: name -> number of categories
def tables():
    t = {}
    for k in range(1, OWN + 1): t[f"own{k}"] = 2 + 2 * (2 * IV + 1)      # unknown/rest, (onset|hold) x interval
    t["degree"] = 24; t["register"] = 10; t["pos"] = 32; t["weight"] = 5; t["role"] = 15
    for s in SLOTS:
        for o in OFFS:
            t[f"{s}{o:+d}"] = 3 + 2 * 12                                # absent, unknown, rest, (onset|hold) x interval class
            t[f"{s}{o:+d}oct"] = 7                                     # none, octave of the interval (-3..+2)
    return t
TABLES = tables()
NAMES = list(TABLES)

def quantise(voice, total):
    """voice: [[onset, dur, midi], ...] in crotchets -> (pitch[T] with -1 for rest, onset[T] bool)."""
    pitch = np.full(total, -1, np.int16); on = np.zeros(total, bool)
    for o, d, m in voice:
        a = int(round(o / STEP)); b = max(a + 1, int(round((o + d) / STEP)))
        if a >= total or on[a]: continue
        b = min(b, total); pitch[a:b] = m; on[a] = True
        on[a + 1:b] = False
    # a later onset inside a held note cuts it (already handled by overwrite order)
    return pitch, on

def bar_info(time_sig):
    num, den = time_sig
    bar = int(round(num * 4 / den / STEP))
    return bar

def weight(pos, bar, num):
    if pos == 0: return 4
    beat = bar // num if num not in (6, 9, 12) else bar // (num // 3)
    if bar % 2 == 0 and pos == bar // 2 and num in (4, 2, 12): return 3
    if beat and pos % beat == 0: return 2
    if pos % 2 == 0: return 1
    return 0

def ival(x, lim): return int(max(-lim, min(lim, x)))

def own_cat(p, onset, p0):
    if p < 0: return 1                                                   # rest (0 = unknown, before the start)
    return 2 + (0 if onset else 2 * IV + 1) + ival(p - p0, IV) + IV

def other_cat(state, p, onset, p0):
    """state: 'absent' | 'unknown' | 'known'. Returns (interval-class category, octave category)."""
    if state == "absent": return 0, 0
    if state == "unknown": return 1, 0
    if p < 0: return 2, 0
    d = p - p0
    return 3 + (0 if onset else 12) + d % 12, 1 + max(-3, min(2, d // 12)) + 3

def target(p, onset, p0):
    if p < 0: return REST
    if not onset: return HOLD
    return ival(p - p0, IV) + IV

def context(grid, v, t, tonic, mode, bar, num, known=None):
    """Feature indices for predicting voice v at step t. grid: list of (pitch, onset) arrays per voice
    (top voice first). known(u, s) -> bool says whether voice u at step s may be seen; default: the
    past of every voice. p0 is v's last sounding pitch before t (None if it has not yet played)."""
    pitch, on = grid[v]
    p0 = None
    for s in range(t - 1, -1, -1):
        if pitch[s] >= 0: p0 = int(pitch[s]); break
    if p0 is None: return None
    f = {}
    for k in range(1, OWN + 1):
        s = t - k
        f[f"own{k}"] = 0 if s < 0 else own_cat(int(pitch[s]), bool(on[s]), p0)
    f["degree"] = ((p0 - tonic) % 12) + (12 if mode == "minor" else 0)
    f["register"] = max(0, min(9, (p0 - (60 + tonic)) // 6 + 5))
    pos = t % bar
    f["pos"] = min(31, pos); f["weight"] = weight(pos, bar, num)
    nv = len(grid); f["role"] = min(14, [0, 0, 1, 3, 6, 10][min(nv, 5)] + v) if nv >= 2 else 0
    above = [u for u in range(v - 1, -1, -1)][:2]; below = [u for u in range(v + 1, nv)][:2]
    for name, lst in (("above", above), ("below", below)):
        for i in range(2):
            u = lst[i] if i < len(lst) else None
            for o in OFFS:
                s = t + o; key = f"{name}{i + 1}{o:+d}"
                if u is None or s < 0 or s >= len(grid[u][0]): f[key], f[key + "oct"] = 0, 0; continue
                vis = (s < t) if known is None else known(u, s)
                f[key], f[key + "oct"] = other_cat("known", int(grid[u][0][s]), bool(grid[u][1][s]), p0) if vis else (1, 0)
    return [f[n] for n in NAMES], p0
