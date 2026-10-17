"""Render a generated fugue to audio (WAV and MP3) with a simple plucked, harpsichord-like synth.

    PYTHONPATH=src python -m fugue.render gen01            # reads results/generated.json
"""
import argparse
import json
import wave
from pathlib import Path

import numpy as np

SR = 44100


def pluck(freq, dur, sr=SR):
    """Additive pluck: bright attack, upper partials decaying faster than the fundamental."""
    n = int((dur + 0.35) * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    for k in range(1, 13):
        f = freq * k * (1 + 0.0004 * k * k)          # slight stiffness
        if f > sr / 2.2:
            break
        decay = 1.2 + 0.9 * k                         # higher partials die sooner
        out += (1 / k ** 0.9) * np.sin(2 * np.pi * f * t) * np.exp(-decay * t)
    env = np.minimum(1, t / 0.003)                    # 3 ms attack
    rel = np.clip((dur + 0.35 - t) / 0.35, 0, 1)      # release after the written length
    return out * env * np.where(t > dur, rel, 1)


def render(notes, tempo=72, sr=SR):
    spq = 60 / tempo
    end = max(on + d for _, on, d, _, _ in notes) * spq + 1.5
    mix = np.zeros(int(end * sr))
    pans = {0: 0.35, 1: 0.6, 2: 0.45, 3: 0.65, 4: 0.5}
    left, right = np.zeros_like(mix), np.zeros_like(mix)
    for v, on, d, m, _ in notes:
        f = 440 * 2 ** ((m - 69) / 12)
        s = pluck(f, d * spq, sr) * 0.12
        i = int(on * spq * sr)
        j = min(len(mix), i + len(s))
        p = pans.get(v, 0.5)
        left[i:j] += s[: j - i] * (1 - p)
        right[i:j] += s[: j - i] * p
    st = np.stack([left, right], axis=1)
    st /= max(1e-9, np.abs(st).max()) / 0.9
    return (st * 32767).astype(np.int16)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--gen", default="results/generated.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tempo", type=int, default=72)
    a = ap.parse_args(argv)
    g = [x for x in json.loads(Path(a.gen).read_text()) if x["piece"] == a.name][0]
    pcm = render(g["notes"], a.tempo)
    out = Path(a.out)
    with wave.open(str(out / f"{a.name}.wav"), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    import lameenc
    enc = lameenc.Encoder()
    enc.set_bit_rate(192)
    enc.set_in_sample_rate(SR)
    enc.set_channels(2)
    enc.set_quality(2)
    mp3 = enc.encode(pcm.tobytes()) + enc.flush()
    (out / f"{a.name}.mp3").write_bytes(mp3)
    print(f"wrote {out / a.name}.mp3 ({len(mp3) / 1e6:.1f} MB, {len(pcm) / SR:.0f} s)")


if __name__ == "__main__":
    main()
