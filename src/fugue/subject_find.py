"""Finding a fugue's subject and its entries in plain note lists (used by scripts/build_open_plans.py and
scripts/prepare_tokens.py)."""
from fugue.coverage import N

MAJ = [0, 2, 4, 5, 7, 9, 11]; MIN = [0, 2, 3, 5, 7, 8, 10]

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

def find_entries(voices, tonic, mode):
    """voices: [[[on, d, midi], ...]] in any order. Returns (entries in order of onset, each with "voice" an index
    into `voices`, subject pattern) or ([], None) if no subject can be found. As in build_open_plans: the subject
    is the longest opening of the first voice that recurs in full in every other voice."""
    sc = MIN if mode == "minor" else MAJ
    vs = [[N(o, d, dia(m, tonic, sc), m) for o, d, m in v] for v in voices if v]
    if len(vs) < 2: return [], None
    order = sorted(range(len(vs)), key=lambda k: vs[k][0].on); vs = [vs[k] for k in order]
    e2 = vs[1][0].on; head = [n for n in vs[0] if n.on < e2 - 1e-9]
    if len(head) < 5: return [], None
    def statements(pat):
        out = []
        for vi, v in enumerate(vs):
            last_end = -1
            for i, tr in tolerant(v, pat):
                if v[i].on < last_end - 1e-9: continue
                k = len(pat); last_end = v[i + k - 1].on + v[i + k - 1].dur
                out.append({"voice": order[vi], "onset": v[i].on, "end": last_end, "transform": tr,
                            "dia_shift": v[i].dia - pat[0].dia, "midi_shift": v[i].midi - pat[0].midi})
        return out
    for L in range(len(head), 4, -1):
        if len(statements(head[:L])) - 1 >= len(vs) - 1:
            ents = statements(head[:L]); ents.sort(key=lambda e: e["onset"]); return ents, head[:L]
    return [], None
