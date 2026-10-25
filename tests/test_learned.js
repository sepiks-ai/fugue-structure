// The browser model must see and predict exactly what the trained Python model does.
// node tests/test_learned.js   (fixture: python scripts/parity_dump.py <model.pt>, after scripts/export_counterpoint.py)
const assert = require("assert"), fs = require("fs"), path = require("path");
const L = require("../web/learned.js");
const meta = JSON.parse(fs.readFileSync(path.join(__dirname, "../web/model/counterpoint.json")));
const buf = fs.readFileSync(path.join(__dirname, "../web/model/counterpoint.bin"));
L.loadNet(meta, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length));
assert.deepStrictEqual(meta.tables, L.TABLES.map(([n, k]) => [n, k]), "feature tables agree");
const fx = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures/parity.json")));
const G0 = { pitch: fx.grid.map((g) => Int16Array.from(g[0])), onset: fx.grid.map((g) => Uint8Array.from(g[1])) };
let worst = 0;
for (const q of fx.queries) {
  const known = G0.pitch.map((p, u) => { const k = new Uint8Array(p.length); for (let s = 0; s < p.length; s++) k[s] = s < q.t || !q.hide[u] ? 1 : 0; return k; });
  const idx = new Int32Array(L.TABLES.length);
  const p0 = L.features({ ...G0, known }, q.v, q.t, fx.tonic, fx.mode, fx.bar, fx.num, idx);
  assert.strictEqual(p0, q.p0);
  assert.deepStrictEqual(Array.from(idx).map((x, f) => x - L.OFFSET[f]), q.feats, `features at v${q.v} t${q.t}`);
  const lg = L.forward(idx);
  for (let k = 0; k < lg.length; k++) worst = Math.max(worst, Math.abs(lg[k] - q.logits[k]));
}
assert.ok(worst < 0.05, "logits agree (int8 embeddings): worst difference " + worst.toFixed(4));
console.log(`learned-model parity passed: ${fx.queries.length} queries, worst logit difference ${worst.toFixed(4)}`);

// ---- the learned generator, end to end
const E = require("../web/engine.js");
E.useLearned(L, JSON.parse(fs.readFileSync(path.join(__dirname, "../web/model/plan_model.json"))));
for (const q of ["A sombre fugue in D minor for three voices", "Bright and lively, four voices, in G major", "a short duet in E minor"]) {
  const t0 = Date.now(), g = E.generate(q, [], 1), ms = Date.now() - t0;
  assert.ok(g.learned, q + ": learned composer used");
  assert.ok(new Set(g.entries.map((e) => e.voice)).size === g.voices, q + ": every voice states the subject");
  assert.deepStrictEqual(E.generate(q, [], 1).notes, g.notes, q + ": reproducible from request and seed");
  for (let v = 0; v < g.voices; v++) {                      // no voice falls silent for long once it has entered
    const ns = g.notes.filter((n) => n.v === v).sort((a, b) => a.on - b.on); let pe = ns[0].on, longest = 0;
    for (const n of ns) { longest = Math.max(longest, n.on - pe); pe = Math.max(pe, n.on + n.d); }
    assert.ok(longest <= 16, `${q}: voice ${v} silent for ${longest} beats`);
  }
  const end = g.notes.filter((n) => n.on >= g.length - 4);
  assert.strictEqual(end.length, g.voices, q + ": one final-chord note per voice");
  assert.ok(ms < 15000, q + ": composed in " + ms + "ms");
  console.log(`learned: ${q} — ${g.voices} voices, ${(g.length / 4).toFixed(0)} bars, ${g.entries.length} entries, ${ms}ms`);
}
console.log("learned generator tests passed");
