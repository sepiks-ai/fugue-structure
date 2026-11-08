"""Training windows from data/tx_pieces.json: a piece is drawn (fugues weighted up), transposed to a random key
it fits, encoded, and a window cut from it starting at a bar line, with the piece's header put in front."""
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
        if k not in s.cache: s.cache[k] = T.encode(s.pieces[i], shift)
        return s.cache[k]
    def sample(s):
        while True:
            i = s.rng.choices(range(len(s.pieces)), s.w)[0]
            ids = s.encoded(i, s.rng.randint(-6, 5))
            if ids: break
        if len(ids) <= s.ctx + 1: seq = ids
        else:
            bars = [j for j in range(HEADER, len(ids) - s.ctx) if ids[j] == T.ID["BAR"]]
            j = s.rng.choice(bars) if bars and s.rng.random() > 0.1 else HEADER
            seq = ids[:HEADER] + ids[j:j + s.ctx + 1 - HEADER]
        return seq + [0] * (s.ctx + 1 - len(seq))
    def batch(s, n):
        x = torch.tensor([s.sample() for _ in range(n)])
        return x[:, :-1], x[:, 1:]
    def tokens(s):
        return sum(len(T.encode(p) or []) for p in s.pieces)
