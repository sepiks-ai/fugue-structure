/* Composes with the three learned layers (tx.js) off the main thread, reporting each layer as it is written. */
importScripts("engine.js", "tx.js");
let ready = null, PM = null;
function load() {
  if (!ready) ready = (async () => {
    const base = new URL(".", self.location.href);
    const get = async (p) => { const r = await fetch(new URL(p, base)); if (!r.ok) throw new Error(p + ": HTTP " + r.status); return r; };
    const [meta, buf, pm] = await Promise.all([get("tx/model.json").then((r) => r.json()), get("tx/model.bin").then((r) => r.arrayBuffer()), get("model/plan_model.json").then((r) => r.json())]);
    FugueTx.init(meta, buf); PM = pm;
  })();
  return ready;
}
self.onmessage = async (ev) => {
  const { text, seed, id } = ev.data;
  try {
    self.postMessage({ id, ev: { stage: "loading" } });
    await load();
    const p = FugueEngine.interpret(text, seed), V = FugueTx.model.vocab;
    const opts = { tonic: p.tonicPc, mode: p.mode, voices: Math.max(2, Math.min(5, p.voices)), seed, motion: p.motion, leap: p.leap,
      length: p.length, inversion: p.inversion, subjects: 6 };
    let lastSent = 0;
    const r = FugueTx.compose(PM, opts, (e) => {
      if (e.stage === "notes") {
        const now = Date.now(); if (now - lastSent < 400) return; lastSent = now;
        self.postMessage({ id, ev: { stage: "notes", notes: e.notes, bar: e.bar } });
      } else self.postMessage({ id, ev: e });
    });
    const ticker = r.seq.slice(-24).map((i) => V[i]);
    const piece = {
      request: text, seed, params: p, key: p.keyName, tempo: p.tempo, timbre: p.timbre, voices: r.voices, layers: true,
      plan: { learned: true, layers: true }, subjectLearned: true,
      subject: r.subject.map((n) => ({ t: n.on, d: n.d, m: n.m })),
      entries: r.entries.map((e) => ({ voice: e.voice, onset: e.onset, end: e.end, inv: e.inv, degree: e.degree })),
      notes: r.notes.sort((a, b) => a.on - b.on || a.v - b.v), length: Math.ceil(r.length), chords: [], learned: true, tokens: r.tokens, ticker,
    };
    self.postMessage({ id, piece });
  } catch (e) {
    self.postMessage({ id, error: String(e && e.message || e) });
  }
};
