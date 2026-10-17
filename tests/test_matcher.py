"""Matcher tests on synthetic voices. No score data needed, so CI can run them."""
from fugue.coverage import N, matches

# A 7-note subject: diatonic steps, mixed rhythm.
SUBJ_DIA = [0, 1, 2, 3, 1, 4, 0]
SUBJ_DUR = [0.5, 0.5, 0.5, 0.75, 0.25, 1.0, 1.0]


def voice(dias, durs, start=0.0, base=28):
    out, t = [], start
    for d, q in zip(dias, durs):
        out.append(N(t, q, base + d, 60))
        t += q
    return out


SUBJ = voice(SUBJ_DIA, SUBJ_DUR)


def full_hits(v):
    return [(i, k, tr) for i, k, tr, full in matches(v, SUBJ) if full]


def test_exact_statement():
    assert full_hits(voice(SUBJ_DIA, SUBJ_DUR)) == [(0, 7, "x1")]


def test_transposed_statement_after_free_material():
    free = voice([0, 5, 2], [1.0, 1.0, 1.0])
    v = free + voice(SUBJ_DIA, SUBJ_DUR, start=3.0, base=32)
    assert full_hits(v) == [(3, 7, "x1")]


def test_inversion():
    v = voice([-d for d in SUBJ_DIA], SUBJ_DUR)
    assert full_hits(v) == [(0, 7, "INV+x1")]


def test_augmentation():
    v = voice(SUBJ_DIA, [2 * q for q in SUBJ_DUR])
    assert full_hits(v) == [(0, 7, "AUG")]


def test_tonal_answer_slack_on_opening_interval():
    dias = SUBJ_DIA.copy()
    dias[1:] = [d - 1 for d in dias[1:]]   # first interval altered, rest intact
    assert full_hits(voice(dias, SUBJ_DUR)) == [(0, 7, "x1~")]


def test_late_interval_change_is_not_a_full_statement():
    dias = SUBJ_DIA.copy()
    dias[5] += 1
    assert full_hits(voice(dias, SUBJ_DUR)) == []


def test_rhythm_change_inside_is_rejected():
    durs = SUBJ_DUR.copy()
    durs[3] = 0.5
    assert full_hits(voice(SUBJ_DIA, durs)) == []
