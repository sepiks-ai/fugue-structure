"""Convert VCSL (CC0) recordings into small web samples: web/samples/<instrument>/<midi>.mp3 + manifest.json.

Pitch comes from the file names, corrected for each set's octave convention (checked against
the audio: harpsichord and organ files name notes an octave lower than scientific pitch).
    python scripts/build_samples.py <VCSL checkout> web/samples
"""
import glob, json, sys, wave
from pathlib import Path
import numpy as np
import lameenc

SR = 44100
SETS = {   # instrument: (globs in order of preference, seconds kept, octave offset of the names)
    "harpsichord": (["Chordophones/Zithers/Harpsichord, Flemish/Sustains/Low/*_rr1.wav",
                     "Chordophones/Zithers/Harpsichord, Flemish/Sustains/High/*_rr1.wav"], 3.2, 12),
    "organ": (["Aerophones/Edge-blown Aerophones/Pipe Organ/Quiet/*_rr1.wav"], 4.5, 12),
    "piano": (["Chordophones/Zithers/Grand Piano, Steinway B/Sus/*_vl3_rr1.wav"], 4.5, 0),
}
NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

def named_midi(path, offset):
    import re
    m = re.search(r"_([A-G])(#?)(-?\d)_(?:vl\d_)?rr", path)
    return 12 * (int(m.group(3)) + 1) + NOTE[m.group(1)] + (1 if m.group(2) else 0) + offset

def read(path):
    w = wave.open(path)
    n, ch, sw = w.getnframes(), w.getnchannels(), w.getsampwidth()
    raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        x = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        x = np.where(x >= 1 << 23, x - (1 << 24), x).astype(np.float32) / (1 << 23)
    else:
        raise ValueError(sw)
    return x.reshape(-1, ch).mean(axis=1), w.getframerate()

def pitch(x, sr):
    """Fundamental by harmonic product spectrum on a sustained window."""
    seg = x[int(0.15 * sr): int(0.15 * sr) + 16384]
    if len(seg) < 4096:
        seg = x[:16384]
    seg = seg * np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg, 1 << 16))
    freqs = np.fft.rfftfreq(1 << 16, 1 / sr)
    hps = spec.copy()
    for k in (2, 3, 4):
        d = spec[::k]
        hps[: len(d)] *= d
    lo, hi = np.searchsorted(freqs, 25), np.searchsorted(freqs, 4500)
    f0 = freqs[lo + np.argmax(hps[lo:hi])]
    return 69 + 12 * np.log2(f0 / 440)

def main(src, out):
    out = Path(out)
    manifest = {}
    for inst, (pats, keep, offset) in SETS.items():
        files = [f for p in pats for f in sorted(glob.glob(str(Path(src) / p)))]
        got = {}
        for f in files:
            x, sr = read(f)
            assert sr == SR
            on = np.argmax(np.abs(x) > 0.02 * np.abs(x).max())
            x = x[max(0, on - 32):]                       # trim the silence before the attack
            midi = named_midi(f, offset)
            if midi in got or midi < 21 or midi > 100:   # first set wins on duplicates
                continue
            n = int(keep * sr)
            x = x[:n]
            fade = int(0.35 * sr)
            x[-fade:] *= np.linspace(1, 0, fade) ** 2
            got[midi] = x
        # even out loudness: match the RMS of the first half second
        lv = {m: np.sqrt(np.mean(x[: SR // 2] ** 2)) for m, x in got.items()}
        target = np.median(list(lv.values()))
        d = out / inst
        d.mkdir(parents=True, exist_ok=True)
        for m, x in got.items():
            y = x * (target / max(lv[m], 1e-6))
            y = y / max(1.0, np.abs(y).max() / 0.95)
            enc = lameenc.Encoder()
            enc.set_bit_rate(96); enc.set_in_sample_rate(SR); enc.set_channels(1); enc.set_quality(2)
            (d / f"{m}.mp3").write_bytes(enc.encode((y * 32767).astype(np.int16).tobytes()) + enc.flush())
        manifest[inst] = sorted(got)
        print(f"{inst}: {len(got)} samples, MIDI {min(got)}-{max(got)}: {sorted(got)}")
    manifest["_licence"] = "Versilian Community Sample Library (VCSL), CC0 1.0. https://versilian-studios.com/vcsl/"
    (out / "manifest.json").write_text(json.dumps(manifest))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
