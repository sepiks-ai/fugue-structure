"""Generator tests: run a fast, low-beam generation on No. 1's plan."""
import json
from pathlib import Path

from fugue.skeleton import fill, parse_subject, realise, to_degree, from_degree

SUBJ = "C4/1 G4/1 A4/.5 G4/.5 F4/.5 E4/.5 D4/.25 E4/.25 F4/.5 E4/.5 D4/.5 C4/.5 B3/.5 C4/1"
PLAN = [p for p in json.loads(Path("results/wtc1_events.json").read_text()) if p["piece"] == "wtc1f01"][0]


def test_degree_roundtrip():
    for m in range(40, 90):
        assert from_degree(*to_degree(m, 60), 60) == m


def test_plan_realised_and_voices_in_range():
    subject = parse_subject(SUBJ)
    placed = realise(subject, PLAN, 60)
    assert len(placed) >= 18
    for p in placed:                                   # no overlapping entries in one voice
        for q in placed:
            if p is not q and p["voice"] == q["voice"]:
                assert p["end"] <= q["onset"] or q["end"] <= p["onset"]
    notes, total, chords, _ = fill(placed, PLAN, 60, beam=6, per_voice=3)
    assert chords[-1] == "I" and chords[-3] == "V7"
    for v, (lo, hi) in enumerate(PLAN["voice_ranges"]):
        ps = [m for vv, on, d, m, src in notes if vv == v and src == "free"]
        assert ps and lo - 2 <= min(ps) and max(ps) <= hi + 2
    assert max(on + d for _, on, d, _, _ in notes) <= total
