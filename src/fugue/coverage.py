"""Copy-and-transform coverage of fugues.

For each fugue: extract voices, infer SUBJECT (first solo statement) and CS1
(first voice's line against the second entry), find every occurrence of each
in every voice under transforms {T, INV} x {1, AUG x2, DIM x0.5}, plus head
fragments, then report the share of notes and of note-duration explained.

Matching follows the principle of Giraud et al. (2012/2015): diatonic interval
similarity, strict duration matching except for the first and last notes.
These are *silver* labels: no hand annotation is used.
"""
import sys, json, math
from dataclasses import dataclass
from pathlib import Path
from music21 import converter

SCALES = {"x1": 1.0, "AUG": 2.0, "DIM": 0.5}


@dataclass
class N:
    on: float      # onset, quarter lengths
    dur: float
    dia: int       # diatonic note number
    midi: int


def voices(path):
    s = converter.parse(str(path))
    out, chords = [], 0
    for p in s.parts:
        notes = []
        for n in p.stripTies().flatten().notes:
            if n.isChord:
                chords += 1
                pt = max(n.pitches, key=lambda x: x.ps)
            else:
                pt = n.pitch
            notes.append(N(float(n.offset), float(n.quarterLength), pt.diatonicNoteNum, int(pt.ps)))
        notes.sort(key=lambda x: x.on)
        if notes:
            out.append(notes)
    out.sort(key=lambda v: v[0].on)          # order voices by first entry
    return out, chords


def lexicon(vs):
    """SUBJECT = first voice up to the second entry; CS1 = first voice during entry 2."""
    v0, e2 = vs[0], vs[1][0].on
    subj = [n for n in v0 if n.on < e2]
    span = subj[-1].on + subj[-1].dur - subj[0].on
    cs = [n for n in v0 if e2 <= n.on < e2 + span]
    lex = {"SUBJECT": subj}
    if len(cs) >= 4:
        lex["CS1"] = cs
    return lex


def ints(seq):
    return [b.dia - a.dia for a, b in zip(seq, seq[1:])]


def prefix_len(voice, i, pat, inv, sc):
    """Longest k such that voice[i:i+k] matches pat[:k]: diatonic intervals equal
    (one mismatch allowed in the first two intervals, for tonal answers), durations
    equal x sc for every note except the first and the k-th (last matched)."""
    pi = ints(pat)
    k, slack = 1, False
    while i + k < len(voice) and k < len(pat):
        a = voice[i + k].dia - voice[i + k - 1].dia
        b = -pi[k - 1] if inv else pi[k - 1]
        if a != b:
            if k - 1 <= 1 and not slack:
                slack = True
            else:
                break
        # note k-1 is now interior (unless it is the first note): check its duration
        if k - 1 >= 1 and abs(voice[i + k - 1].dur - pat[k - 1].dur * sc) > 1e-6:
            break
        k += 1
    return k, slack


def matches(voice, pat, full_frac=0.75, frag_min=5):
    """Yield (start, length, transform, is_full). Full statement: prefix covers
    >= full_frac of pat (all of it if pat has <= 6 notes). Fragment: >= frag_min notes."""
    L = len(pat)
    need_full = L if L <= 6 else max(frag_min, math.ceil(full_frac * L))
    for i in range(len(voice)):
        best = None
        for inv in (False, True):
            for tag, sc in SCALES.items():
                if abs(voice[i].dur - pat[0].dur * sc) > 1e-6 and sc != 1.0:
                    continue        # aug/dim: require the scaled rhythm from the start
                k, sl = prefix_len(voice, i, pat, inv, sc)
                if best is None or k > best[0]:
                    best = (k, ("INV+" if inv else "") + tag + ("~" if sl else ""))
        k, tr = best
        if k >= need_full:
            yield i, k, tr, True
        elif k >= frag_min:
            yield i, k, tr, False


def analyse(path):
    vs, chords = voices(path)
    if len(vs) < 2:
        return None
    lex = lexicon(vs)
    covered = {}                   # (voice, idx) -> label
    events = []
    cands = []
    for name, pat in lex.items():
        for vi, v in enumerate(vs):
            for i, k, tr, full in matches(v, pat):
                cands.append((full, k, name if full else f"FRAG({name})", vi, i, tr))
    # full statements before fragments, longer before shorter; first claim wins
    for full, k, name, vi, i, tr in sorted(cands, key=lambda c: (not c[0], -c[1])):
        idx = [(vi, j) for j in range(i, i + k)]
        if any(x in covered for x in idx):
            continue
        for x in idx:
            covered[x] = name
        last = vs[vi][i + k - 1]
        events.append({"material": name, "voice": vi, "onset": vs[vi][i].on,
                       "end": last.on + last.dur, "n": k, "transform": tr})
    tot_n = sum(len(v) for v in vs)
    tot_d = sum(n.dur for v in vs for n in v)

    def share(pred):
        n = sum(1 for x, lab in covered.items() if pred(lab))
        d = sum(vs[x[0]][x[1]].dur for x, lab in covered.items() if pred(lab))
        return n / tot_n, d / tot_d

    return {
        "piece": Path(path).stem, "voices": len(vs),
        "length": max(v[-1].on + v[-1].dur for v in vs),
        "voice_entries": [v[0].on for v in vs], "notes": tot_n, "chords_in_voices": chords,
        "subject_notes": len(lex["SUBJECT"]), "cs_notes": len(lex.get("CS1", [])),
        "full_entries": sum(e["material"] == "SUBJECT" for e in events),
        "cs_entries": sum(e["material"] == "CS1" for e in events),
        "cov_subject": share(lambda l: l == "SUBJECT"),
        "cov_subj_cs": share(lambda l: l in ("SUBJECT", "CS1")),
        "cov_all": share(lambda l: True),
        "events": events,
    }


if __name__ == "__main__":
    files = sorted(Path(sys.argv[1]).glob(sys.argv[2]))
    res = [r for r in (analyse(f) for f in files) if r]
    Path(sys.argv[3]).write_text(json.dumps(res, indent=1))
    print(f"{'piece':10} v  notes subjN  S#  CS#   S%n   S+CS%n  all%n   all%dur")
    for r in res:
        print(f"{r['piece']:10} {r['voices']} {r['notes']:5} {r['subject_notes']:5} {r['full_entries']:3} {r['cs_entries']:4}"
              f"  {100*r['cov_subject'][0]:5.1f}  {100*r['cov_subj_cs'][0]:6.1f}  {100*r['cov_all'][0]:5.1f}  {100*r['cov_all'][1]:7.1f}")
    import statistics as st
    for key in ("cov_subject", "cov_subj_cs", "cov_all"):
        xs = [r[key][0] for r in res]
        print(f"{key:12} notes: median {100*st.median(xs):.1f}%  mean {100*st.mean(xs):.1f}%  range {100*min(xs):.1f}-{100*max(xs):.1f}%")
