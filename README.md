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

## Write a Fugue (in the browser)

`web/compose.html`: a reader describes a fugue in plain words ("a sombre fugue
in D minor for three voices") and it is composed and played in the browser, with
MIDI and WAV downloads and a shareable link that reproduces the piece. Live at
https://sepiks-ai.github.io/fugue-structure/compose.html once Pages is enabled.

- `web/engine.js`: request interpreter, subject composer, plan realisation with
  local-key transposition (major and minor), countersubject, episodes, harmony,
  beam-search counterpoint, MIDI writer
  - the countersubject is composed once against the subject by dynamic programming:
    thirds, sixths and octaves on the beat (so it inverts at the octave), rhythm that
    moves where the subject holds, dissonance only as passing notes or prepared
    suspensions; it returns against later entries in a free voice
  - episodes fill the gaps between groups of entries with a head or tail fragment of the
    subject or countersubject in a diatonic sequence, imitated by a second voice
- `web/worker.js`: runs the engine off the main thread
- `results/plans.json`: compact Bach entry plans (`scripts/build_plans.py`)
- `tests/test_engine.js`: interpretation, determinism, dissonance treatment, countersubject and parallels

## Skeleton fugue generator

`src/fugue/skeleton.py` writes a fugue on a new subject in four steps:

1. **Plan**: take the subject-entry grid of a WTC fugue (voice, onset, pitch level, transform).
2. **Realise**: place transformed copies of the new subject on that grid, time-scaled
   to its length; entries that clash in stretto are nudged or dropped.
3. **Harmony**: one chord per half-bar, chosen by Viterbi to fit the fixed entries.
4. **Fill**: beam search over quavers for the free voices (chord tones on beats,
   passing notes between, no parallel fifths or octaves, cadence at the end).

```bash
PYTHONPATH=src python -m fugue.skeleton --name gen01 \
  --subject "C4/1 G4/1 A4/.5 G4/.5 F4/.5 E4/.5 D4/.25 E4/.25 F4/.5 E4/.5 D4/.5 C4/.5 B3/.5 C4/1"
```

Then `PYTHONPATH=src python -m fugue.render gen01` writes `results/gen01.mp3`.

Writes `results/gen01.mid` and adds the piece to `results/generated.json`, which
the Fugue Map lists under *Generated* with a Play button. Notes are written as
`name+octave/length in crotchets`. Options: `--key G`, `--template wtc1f05`, `--seed 3`.

Current limits: major keys only; entries off the tonic, fourth or fifth are
transposed diatonically in the home key; no countersubject; episodes are free
counterpoint with no motivic material; rhythm of free voices is mostly quavers.

## Fugue Map

`web/index.html` is an interactive map of every entry, voice by voice, for
each WTC I fugue and each generated fugue. It is deployed to GitHub Pages on
every push to `main`.

To view it locally on Windows, double-click **`run.bat`**. It starts a local
server and opens http://localhost:8000/web/. Close the window to stop it.
Elsewhere:

```bash
python -m http.server 8000      # from the repo root
# open http://localhost:8000/web/
```

## The learned composer

The page now composes with two models learned from the 48 fugues of the *Well-Tempered Clavier*,
using only openly licensed data: Kyle Rother's open edition (2015, CC BY 4.0) [5], typeset in LilyPond
from the Bach-Gesellschaft text. Huron's encoding [1] is used only to check the conversion, never to train.

**Data** (`scripts/open_wtc.py` → `data/open_wtc.json`): LilyPond renders each staff (one voice per staff
in this edition) to MIDI; 48 fugues, 166 voices, 50,929 notes. Against Huron: same number of voices in
every fugue, median note-for-note agreement 98.9%; the rest are editorial readings of rhythm, plus one
octave slip in the edition (Book I No. 4, mezzo from bar 25), corrected and recorded in the script.

**Counterpoint model** (`src/fugue/learn/`, `scripts/train_counterpoint.py`): on a semiquaver grid, each
voice at each step starts a note (an interval from its last pitch), holds or rests. A small network
(an embedding bag of 81 categorical features, two layers of 128; 347 KB exported) predicts that step from
the voice's last three crotchets, two voices above and two below a crotchet either side of now (any of which
may be "not yet written"), the beat, the scale degree and the voice's role. On six held-out fugues it reaches
a loss of 1.00 nats per step (perplexity 2.7), against 1.26 for a melody-only model of the same voice and
1.85 for frequencies alone: the other voices account for about a quarter of the remaining uncertainty.

**Plan model** (`scripts/build_open_plans.py`, `scripts/learn_plans.py`, `scripts/learn_subjects.py` → `results/plan_model.json`):
entries are found with a tolerant matcher (85% of intervals and rhythm), the subject being the longest opening
of the first voice that recurs. From the 48 plans it estimates, with Dirichlet smoothing, the exposition order,
the key of each later entry given the previous one and the position in the piece, the time between entries
(so stretto), inversion and augmentation, the voice, the length, and when to stop.

**Subject** (`learnedSubject` in `web/engine.js`, `scripts/learn_subjects.py`): the counterpoint model plays the
opening of a fugue alone, as Bach's first voice does, 48 times over (pitch choices sharpened to temperature 0.75).
Each line is cut where a phrase can end (on a beat, on a note of the tonic triad, near a length drawn from the
48 subjects), and the one kept scores best on the model's own confidence plus its likeness to Bach's subjects:
seven measures (notes per crotchet, range, share of steps, share of leaps, number of note lengths, longest note,
where the highest note falls), each with a mean and spread estimated from the 48. The description moves the
targets (lively asks for more notes, angular for more leaps, chromatic welcomes altered notes).

**Generation** (`web/learned.js`): eight drafts sample the model step by step, lowest voice first, then each
voice reconsiders knowing the others' choices (a Gibbs sweep); every bar the drafts with the most probable
pitch choices are kept (rhythm is left to the model, since ranking by full likelihood favours held notes).
Each step is two decisions: whether to move (a new note, a hold or a rest), then which note. Three settings were
chosen by measuring the results against Bach's fugues: pitch temperature 0.85 with nucleus 0.95; the odds of
moving raised by e^0.6 between beats (the model alone holds notes more often than Bach does); and on the beat, no
leap into a second, tritone or seventh against a sounding voice. The line the model writes against the first
answer becomes the countersubject. Also imposed: voice ranges, a few texture limits (how many voices may rest at
once, how long a rest or a held note may last), subject-fragment sequences in episodes, and the final chord.

On the beat, compared with Bach's own fugues (same measure, 12 pieces): suspensions 8.5% of voice pairs (Bach
8.7%), untreated dissonance 1.3% (1.1%), all dissonance 25.7% (22.6%); notes of a quaver or shorter, 77% (79%).
Before these settings: suspensions 7.5%, untreated 2.2%, quick notes about 65%. The earlier rule-based composer
gave 3–8% dissonance in all.
`node tests/test_learned.js` checks that the browser computes exactly what the trained Python model does.

## The token composer (tokens all the way down)

The learned composer above writes notes inside a structure built by hand. The token composer has no
scaffold: one transformer writes a whole piece as a single sequence of tokens, structure included.

**Tokens** (`src/fugue/tx/tokens.py`, 434 of them). A header (form, key, meter, number of voices), then bar
lines, positions within the bar (12 per crotchet, so triplets fit), and notes as voice, interval, length.
Pitches are absolute, except inside a marked subject statement, where each note after the first is the
interval from the one before: a subject restated at another pitch is then the same run of tokens, and stating
it again is copying. Each note's absolute pitch is also fed to the model alongside its token. (Run 2,
absolute pitches throughout: convincing local writing but no recurrence, 0.002 against Bach's 0.025 to 0.056.
Run 3, intervals throughout: recurrence 0.013, but from a repeated-note figure looping in 32 bars, two of
three voices, and mostly quavers.)
Fugue entries are tokens too: `ENTRY ROLE_x1 SHIFT_7 VOICE_2` says the subject begins here in voice 2, a fifth
up. The notes after it are written like any others, so stating the subject correctly is the model's job.
Notes struck together are written lowest voice first.

**Data** (`scripts/mutopia_bach.py`, `scripts/prepare_tokens.py`, `data/OPEN_BACH.md`): Rother's 48 fugues plus
every Bach piece in the Mutopia Project with a Public Domain, CC BY or CC BY-SA licence (WTC copies left out),
rendered by LilyPond with a MIDI track per voice; fragments of voices are joined, orchestral scores (more than
12 tracks) and duplicates dropped. 590 pieces, 1.7 million tokens, of which 80 fugues (the 48, *The Art of
Fugue*, organ and keyboard fugues). Entries are marked in 75 of the fugues and 6 inventions by the same
subject finder as the plans (`src/fugue/subject_find.py`). Keys of Mutopia pieces are estimated (the
estimator gets 42 of the 48 WTC keys right). Every piece is transposed into random keys during training.

**Model** (`src/fugue/tx/model.py`): decoder-only transformer, 8 layers, width 384, 6 heads, rotary positions,
about 14 million parameters, context 2048 tokens (about half a WTC fugue). Training windows start at bar
lines with the header kept in front; fugues are drawn four times as often as their size would suggest.

**Generation** (`scripts/sample_tx.py`): given only the header, the model writes bars, entries, notes and the
end. A grammar keeps the tokens well formed (a voice is followed by a pitch, a pitch by a length; positions
move forward within a bar; no more voices than asked for). Nothing else is imposed.

### Training on Windows with two GTX 1080 Ti

PyTorch 2.14 is the last release whose CUDA 12.6 wheels include Pascal cards [6].

```
git clone https://github.com/sepiks-ai/fugue-structure
cd fugue-structure
py -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
py -m pip install mido
py scripts\train_tx.py
```

It uses both cards (DataParallel), trains in float32 (Pascal is slow at half precision), prints progress
every 50 steps and held-out loss every 500, and keeps `results\tx\best.pt` and `results\tx\last.pt`
(`--resume` continues). Then:

```
py scripts\sample_tx.py --form=FUGUE --key=2_minor --meter=4/4 --voices=3 --seed=1
```

writes `results\tx\sample.mid` and `.json`.

## Reproduce

```bash
pip install -r requirements.txt
scripts/fetch_data.sh
PYTHONPATH=src python -m fugue.coverage data/bach-wtc/kern 'wtc1f*.krn' results/wtc1_events.json
PYTHONPATH=src python -m fugue.null_baseline data/bach-wtc/kern 'wtc1f*.krn' results/wtc1_null.json
PYTHONPATH=src pytest -q tests     # matcher tests; no score data needed
# the learned composer (open data; needs lilypond, mido, torch; see the section above)
python scripts/open_wtc.py <midi_dir> <rother_ly_root> data/open_wtc.json
python scripts/build_open_plans.py && python scripts/learn_plans.py && python scripts/learn_subjects.py
python scripts/train_counterpoint.py            # six held-out fugues, reports held-out loss
python scripts/train_counterpoint.py --all --epochs=3 && python scripts/export_counterpoint.py
node tests/test_learned.js
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
5. Rother, K. (2015). An "open source" open score edition of Bach's Well-tempered Clavier (48 fugues). OpenUCT, University of Cape Town. CC BY 4.0. https://open.uct.ac.za/items/d2a6887b-7196-4fbe-8b56-a469bf557f93
6. PyTorch (2026). Notice: CUDA 12.6 wheels will no longer be published from PyTorch 2.15 (drops Maxwell / Pascal / Volta). dev-discuss.pytorch.org/t/3432
7. The Mutopia Project. Bach, J. S., engraved sources. https://www.mutopiaproject.org/ ; github.com/MutopiaProject/MutopiaProject

## Sound

Recordings from the Versilian Community Sample Library (CC0 1.0,
https://versilian-studios.com/vcsl/), chosen by measured signal-to-noise and evenness: the harpsichord
VCSL calls "Unk" (sampled at every semitone from F1 to F6, 104 dB, even in brightness from bass to treble),
played with its recorded key releases; an English harpsichord's lute stop; an open-diapason organ stop (60 dB)
and a soft Renaissance flute stop; and a Steinway B. `scripts/build_samples.py` converts them
(high-pass at 35 Hz, each note ends where it nears its own noise floor, quiet samples
lifted by at most 6 dB). Strings remain synthesised. The room is a recorded one, convolved: St Andrew's Church
for harpsichord, piano and strings, the Lady Chapel of St Albans Cathedral for organ (impulse responses from
the OpenAIR library, University of York, via Reverb.js; CC BY-SA 3.0, see `web/rooms/LICENCE.md`), matched in
loudness to the synthetic room that remains as a fallback. Playback softens the harpsichord's treble a little,
spreads the voices across the stereo field, brings out the subject, varies timing slightly and slows
into the final cadence. Readers can switch instrument, tempo and touch while it plays.
