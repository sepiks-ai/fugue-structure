"""Convert VCSL (CC0) recordings into small web samples: web/samples/<instrument>/<midi>.mp3 + manifest.json.

Pitch comes from the file names, with each set's octave convention found by measuring the audio.
Processing removes noise rather than amplifying it: high-pass at 35 Hz, each note ends where it
sinks to 12 dB above its own noise floor, and quiet samples are lifted by at most 6 dB.
    python scripts/build_samples.py <VCSL checkout> web/samples
"""
import glob, json, sys, wave
from pathlib import Path
import numpy as np
import lameenc

SR = 44100
SETS = {   # output name: (globs in order of preference, seconds kept)
    # English (Zuckermann kit) harpsichord: 105 dB signal-to-noise; the Flemish set was 53-73 dB
    "harpsichord": (["Chordophones/Zithers/Harpsichord, English/Sustains/Normal/*_rr1.wav"], 3.2),
    "harpsichord_lute": (["Chordophones/Zithers/Harpsichord, English/Sustains/Lute/*_rr1.wav"], 2.6),
    # open diapason (principal): 60 dB; the 'Quiet' stop was 44 dB with audible wind noise
    "organ": (["Aerophones/Edge-blown Aerophones/Pipe Organ/Loud/*.wav"], 4.5),
    "organ_soft": (["Aerophones/Edge-blown Aerophones/Renaissance Organ/8'/*_rr1.wav"], 4.5),
    "piano": (["Chordophones/Zithers/Grand Piano, Steinway B/Sus/*_vl3_rr1.wav"], 4.5),
}
NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

def named(path):
    import re
    m = re.search(r"_([A-G])(#?)(-?\d)(?:_|\.wav)", path.split("/")[-1])
    return 12 * (int(m.group(3)) + 1) + NOTE[m.group(1)] + (1 if m.group(2) else 0)

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

def highpass(x, sr, fc=35.0):
    a = np.exp(-2 * np.pi * fc / sr)
    y = np.empty_like(x); prev_x = prev_y = 0.0
    for i, v in enumerate(x):                       # one-pole high-pass: removes rumble and DC
        prev_y = a * (prev_y + v - prev_x); prev_x = v; y[i] = prev_y
    return y

def octave_offset(files):
    """Each set names notes with its own octave convention: find it by comparing names with measured pitch."""
    votes = []
    for f in files[len(files) // 4: 3 * len(files) // 4]:  # mid-range notes measure reliably
        x, sr = read(f)
        on = np.argmax(np.abs(x) > 0.02 * np.abs(x).max())
        votes.append(12 * round((pitch(x[on:], sr) - named(f)) / 12))
    return max(set(votes), key=votes.count)

def main(src, out):
    out = Path(out)
    manifest = {}
    for inst, (pats, keep) in SETS.items():
        files = [f for p in pats for f in sorted(glob.glob(str(Path(src) / p)))]
        off = octave_offset(files)
        got = {}
        for f in files:
            midi = named(f) + off
            if midi in got or midi < 21 or midi > 100:
                continue
            x, sr = read(f)
            assert sr == SR
            w = int(0.1 * sr)                                              # noise floor: the quietest 100 ms
            frames = x[: len(x) // w * w].reshape(-1, w)
            noise = np.sqrt((frames ** 2).mean(axis=1)).min() + 1e-7
            on = np.argmax(np.abs(x) > 0.02 * np.abs(x).max())
            x = highpass(x[max(0, on - 32):][: int(keep * sr)].astype(np.float64), sr)
            # end the note where it sinks to 12 dB above its own noise, so hiss is never sustained
            hop = 1024
            rms = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, mode="same"))
            below = np.where(rms[int(0.3 * sr):] < 4 * noise)[0]
            end = int(0.3 * sr) + below[0] if len(below) else len(x)
            x = x[:end]
            fade = min(int(0.3 * sr), len(x) // 3)
            x[-fade:] *= np.linspace(1, 0, fade) ** 2
            got[midi] = x
        lv = {m: np.sqrt(np.mean(x[: SR // 2] ** 2)) for m, x in got.items()}
        target = np.median(list(lv.values()))
        d = out / inst
        d.mkdir(parents=True, exist_ok=True)
        for m, x in got.items():
            g = min(2.0, target / max(lv[m], 1e-6))          # never lift a quiet sample by more than 6 dB
            y = x * g
            y = y / max(1.0, np.abs(y).max() / 0.95)
            enc = lameenc.Encoder()
            enc.set_bit_rate(112); enc.set_in_sample_rate(SR); enc.set_channels(1); enc.set_quality(2)
            (d / f"{m}.mp3").write_bytes(enc.encode((y * 32767).astype(np.int16).tobytes()) + enc.flush())
        manifest[inst] = sorted(got)
        print(f"{inst}: {len(got)} samples, MIDI {min(got)}-{max(got)}, octave offset {off:+d}")
    manifest["_licence"] = "Versilian Community Sample Library (VCSL), CC0 1.0. https://versilian-studios.com/vcsl/"
    (out / "manifest.json").write_text(json.dumps(manifest))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
