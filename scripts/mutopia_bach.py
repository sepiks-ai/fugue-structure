"""Open Bach beyond the 48: every Bach piece in the Mutopia Project whose licence is Public Domain, CC BY or
CC BY-SA, rendered by LilyPond to MIDI with one track per voice, read back as notes.

    python scripts/mutopia_bach.py <MutopiaProject/ftp/BachJS> data/open_bach.json

The Well-Tempered Clavier (BWV 846-893) is left out: it comes from Rother's edition (data/open_wtc.json), and
keeping Mutopia's copies out keeps the held-out fugues unseen. Output (CC BY-SA 4.0, see data/OPEN_BACH.md):
{"pieces": [{"id", "bwv", "title", "licence", "source", "meters": [[onset, num, den]], "key": [sharps, minor],
"voices": [[[onset, duration, midi], ...], ...]}]} with times in crotchets, voices top first."""
import json, os, re, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
import mido, lilypond

BIN = os.path.dirname(lilypond.executable())
OPEN = re.compile(r"public domain|creative commons attribution(-sharealike)? [0-9.]+", re.I)
PER_VOICE = r'''
\midi { \context { \Staff \remove "Staff_performer" } \context { \Voice \consists "Staff_performer" } }
'''

def is_wtc(d):
    m = re.match(r"BWV(\d+)", d)
    return bool(m) and 846 <= int(m.group(1)) <= 893

def read_midi(path):
    mf = mido.MidiFile(path); tpq = mf.ticks_per_beat
    meters, key, voices = [], None, []
    for tr in mf.tracks:
        t, on, notes = 0, {}, []
        for msg in tr:
            t += msg.time
            if msg.type == "time_signature": meters.append([round(t / tpq, 4), msg.numerator, msg.denominator])
            elif msg.type == "key_signature" and key is None:
                k = msg.key; minor = k.endswith("m"); key = [k.rstrip("m"), minor]
            elif msg.type == "note_on" and msg.velocity > 0: on.setdefault(msg.note, []).append(t)
            elif msg.type in ("note_off", "note_on") and on.get(msg.note):
                s = on[msg.note].pop(0)
                if t > s: notes.append([round(s / tpq, 4), round((t - s) / tpq, 4), msg.note])
        if notes: voices.append(sorted(notes))
    voices.sort(key=lambda v: -sum(n[2] for n in v) / len(v))
    meters = sorted({tuple(m) for m in meters})
    return [list(m) for m in meters], key, voices

def render(job):
    d, src, work = job
    out = []
    mains = [os.path.join(r, f) for r, _, fs in os.walk(work) for f in fs if f.endswith(".ly") and "\\score" in open(os.path.join(r, f), errors="ignore").read()]
    for ly in mains:
        txt = open(ly, errors="ignore").read()
        txt = re.sub(r'(\\version\s+"[^"]*")', r"\1" + PER_VOICE.replace("\\", "\\\\"), txt, count=1) if "\\version" in txt else PER_VOICE + txt
        open(ly, "w").write(txt)
        r = subprocess.run([os.path.join(BIN, "lilypond"), "--loglevel=ERROR", "-dno-print-pages", "-dmidi-extension=mid", "-o", os.path.splitext(ly)[0], ly],
                           cwd=os.path.dirname(ly), capture_output=True, text=True, timeout=600)
        mids = sorted(f for f in os.listdir(os.path.dirname(ly)) if f.startswith(os.path.basename(os.path.splitext(ly)[0])) and f.endswith(".mid"))
        if not mids: out.append(("fail", ly, r.stderr[-300:])); continue
        for m in mids:
            out.append(("ok", ly, os.path.join(os.path.dirname(ly), m)))
    return d, src, out

def main(root, dest):
    tmp = tempfile.mkdtemp(prefix="mutopia-"); jobs = []
    for d in sorted(os.listdir(root)):
        if is_wtc(d): continue
        files = [os.path.join(r, f) for r, _, fs in os.walk(os.path.join(root, d)) for f in fs if f.endswith((".ly", ".ily"))]
        text = "".join(open(f, errors="ignore").read() for f in files)
        lic = re.search(r'license\s*=\s*"([^"]+)"', text) or re.search(r'copyright\s*=\s*"([^"]+)"', text)
        if not lic or not OPEN.search(lic.group(1)) or "noncommercial" in lic.group(1).lower(): print("skip (licence)", d); continue
        title = re.search(r'\btitle\s*=\s*"([^"]+)"', text); src = {"licence": OPEN.search(lic.group(1)).group(0), "title": title.group(1) if title else d}
        work = os.path.join(tmp, d); shutil.copytree(os.path.join(root, d), work, symlinks=True, ignore=lambda dd, fs: [f for f in fs if not (f.endswith(('.ly', '.ily')) or os.path.isdir(os.path.join(dd, f)))])
        lyf = [os.path.join(r, f) for r, _, fs in os.walk(work) for f in fs if f.endswith((".ly", ".ily"))]
        subprocess.run([os.path.join(BIN, "convert-ly"), "-e", "-d", *lyf], capture_output=True)
        jobs.append((d, src, work))
    pieces, fails = [], []
    with ThreadPoolExecutor(max_workers=int(os.environ.get("JOBS", 2))) as ex:
        for d, src, out in ex.map(render, jobs):
            for status, ly, x in out:
                if status != "ok": fails.append((d, x)); continue
                meters, key, voices = read_midi(x)
                if not voices or sum(len(v) for v in voices) < 30: continue
                pieces.append({"id": d + "/" + os.path.splitext(os.path.basename(x))[0], "bwv": d, "title": src["title"], "licence": src["licence"],
                               "source": "https://www.mutopiaproject.org/ftp/BachJS/" + d, "meters": meters, "key": key, "voices": voices})
            print(d, len(out), "movements", flush=True)
    json.dump({"pieces": pieces}, open(dest, "w"), separators=(",", ":"))
    print(len(pieces), "pieces;", sum(len(v) for p in pieces for v in p["voices"]), "notes;", len(fails), "failed:", [f[0] for f in fails])

if __name__ == "__main__":
    main(*sys.argv[1:3])
