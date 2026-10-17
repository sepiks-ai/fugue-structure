"""Chance baseline: coverage of fugue i by the lexicon (SUBJECT, CS1) of fugue j != i.
Same matcher, same thresholds. If own-lexicon coverage is not well above this,
the coverage figure measures scalar filler, not referenced material."""
import sys, json, statistics as st
from pathlib import Path
from fugue.coverage import voices, lexicon, matches

files = sorted(Path(sys.argv[1]).glob(sys.argv[2]))
data = {f.stem: voices(f)[0] for f in files}
data = {k: v for k, v in data.items() if len(v) >= 2}
lex = {k: lexicon(v) for k, v in data.items()}


def coverage(vs, lx, full_only=False):
    cands = []
    for name, pat in lx.items():
        for vi, v in enumerate(vs):
            for i, k, tr, full in matches(v, pat):
                if full or not full_only:
                    cands.append((full, k, vi, i))
    cov = set()
    for full, k, vi, i in sorted(cands, key=lambda c: (not c[0], -c[1])):
        idx = {(vi, j) for j in range(i, i + k)}
        if idx & cov:
            continue
        cov |= idx
    return len(cov) / sum(len(v) for v in vs)


rows = []
for a in data:
    own = coverage(data[a], lex[a])
    null = [coverage(data[a], lex[b]) for b in data if b != a]
    rows.append((a, own, st.mean(null), max(null)))
    print(f"{a:10} own {100*own:5.1f}%   null mean {100*st.mean(null):5.1f}%  null max {100*max(null):5.1f}%")
print(f"MEDIAN own {100*st.median(r[1] for r in rows):.1f}%  null-mean {100*st.median(r[2] for r in rows):.1f}%")
print(f"excess (own - null mean): median {100*st.median(r[1]-r[2] for r in rows):.1f} pts")
Path(sys.argv[3]).write_text(json.dumps(rows, indent=1))

print("\n-- full statements only (SUBJECT, CS1; no fragments) --")
fo = [(a, coverage(data[a], lex[a], True), st.mean(coverage(data[a], lex[b], True) for b in data if b != a)) for a in data]
print(f"MEDIAN own {100*st.median(r[1] for r in fo):.1f}%  null-mean {100*st.median(r[2] for r in fo):.1f}%  "
      f"excess median {100*st.median(r[1]-r[2] for r in fo):.1f} pts")
