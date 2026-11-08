"""Training windows from data/tx_pieces.json: a piece is drawn (fugues weighted up), transposed to a random key
it fits, encoded, and a window cut from it starting at a bar line, with the piece's header put in front.

A window cut from the middle carries a CONT token after the header. Without it, "header, bar line" was most
often followed by the middle of a piece, so a piece generated from the header began as if halfway through:
all voices at once, in quavers, a texture it then kept (the first run wrote no semiquavers at all, though
it predicts Bach's held-out rhythms well: 30% semiquavers against his 31%). With it, the header alone means
the true beginning."""
import json, random, torch
from . import tokens as T

HEADER = 5
FORM_WEIGHT = {"FUGUE": 4.0, "INVENTION": 2.0, "CHORALE": 1.0, "OTHER": 1.0}

class Windows:
    def __init__(s, path, split, ctx, seed=0):
        s.pieces = [p for p in json.load(open(path))["pieces"] if p["split"] == split]
        s.ctx, s.rng, s.cache = ctx, random.Random(seed), {}
        s.w = [FORM_WEIGHT[p["form"]] * sum(len(v) for v in p["voices"]) ** 0.5 for p in s.pieces]
    def encoded(s, i, shift):
        k = (i, shift)
        if k not in s.cache:
            ids = T.encode(s.pieces[i], shift)
            s.cache[k] = (ids, T.aux_of(ids)) if ids else None
        return s.cache[k]
    def sample(s):
        while True:
            i = s.rng.choices(range(len(s.pieces)), s.w)[0]
            e = s.encoded(i, s.rng.randint(-6, 5))
            if e: break
        ids, aux = e
        if len(ids) <= s.ctx + 1: seq, sa = ids, aux
        else:
            bars = [j for j in range(HEADER, len(ids) - s.ctx) if ids[j] == T.ID["BAR"]]
            j = s.rng.choice(bars) if bars and s.rng.random() > 0.25 else HEADER
            if j == HEADER: seq, sa = ids[:s.ctx + 1], aux[:s.ctx + 1]
            else:
                seq = ids[:HEADER] + [T.ID["CONT"]] + ids[j:j + s.ctx - HEADER]
                sa = aux[:HEADER] + [0] + aux[j:j + s.ctx - HEADER]
        pad = s.ctx + 1 - len(seq)
        return seq + [0] * pad, sa + [0] * pad
    def batch(s, n):
        b = [s.sample() for _ in range(n)]
        x = torch.tensor([q for q, _ in b]); a = torch.tensor([r for _, r in b])
        return x[:, :-1], a[:, :-1], x[:, 1:]
    def tokens(s):
        return sum(len(T.encode(p) or []) for p in s.pieces)
