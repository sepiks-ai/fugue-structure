# fugue-structure

Hierarchical generation of fugues with explicit long-range structure.

The premise: long-range dependency in a fugue is mostly *identity* — the same
subject and countersubject returning, transformed — so the upper levels of the
model should hold pointers to earlier material plus transformation operators,
not category labels.

## Representation

| Level | Content |
|---|---|
| L0 Plan | key, voices, section sequence (EXPO, EPISODE, MIDDLE_ENTRY, STRETTO, CODA) |
| L1 Lexicon | SUBJECT, ANSWER, CS1, CS2 … as short note sequences |
| L2 Event grid | ENTRY(material, voice, onset, transform), SEQUENCE, CADENCE, PEDAL |
| L3 Surface | notes in every voice |

Transforms: diatonic transposition, inversion, augmentation, diminution,
fragmentation, and compositions thereof.

## First result: how much of a Bach fugue is referenced material?

WTC Book I, 24 fugues, silver labels (lexicon inferred, no hand annotation).
Share of notes accounted for by copy-and-transform of the fugue's own
SUBJECT and CS1, against a cross-fugue chance baseline (fugue *i* matched
with the lexicon of fugue *j ≠ i*).

| Matched material | Own (median) | Chance (median) |
|---|---|---|
| Complete statements only | 22.3% | 0.0% |
| Statements + head fragments | 46.8% | 13.1% |

Above-chance referenced material: about 34% of notes.

Known biases: second/third subjects not modelled (No. 4 is a triple fugue);
inferred subject ends overshoot in some fugues (Nos. 7, 21); episodic
sequences not yet encoded; matching is diatonic, accidentals unchecked.

## Reproduce

```bash
pip install -r requirements.txt
scripts/fetch_data.sh
PYTHONPATH=src python -m fugue.coverage data/bach-wtc/kern 'wtc1f*.krn' results/wtc1_events.json
PYTHONPATH=src python -m fugue.null_baseline data/bach-wtc/kern 'wtc1f*.krn' results/wtc1_null.json
```

## Next

1. Validate against the Algomus WTC ground truth [3]; replace inferred subject ends.
2. Encode episodes (SEQUENCE) and additional subjects.
3. Extend to WTC II and *Die Kunst der Fuge*.

## References

1. Huron, D. (1994). *Das wohltemperirte Clavier*, Humdrum **kern encoding. humdrum-tools/bach-wtc.
2. Giraud, M., Groult, R. & Levé, F. (2012). Subject and Counter-Subject Detection for Analysis of the Well-Tempered Clavier Fugues. HAL: hal-00712554.
3. Giraud, M., Groult, R., Leguy, E. & Levé, F. (2015). Computational Fugue Analysis. *Computer Music Journal*, 39(2), 77–96.
4. Wang, Z., Min, L. & Xia, G. (2024). Whole-Song Hierarchical Generation of Symbolic Music Using Cascaded Diffusion Models. *ICLR 2024*. arXiv:2405.09901.
