// Engine tests for CI: node tests/test_engine.js
const assert = require("assert");
const E = require("../web/engine.js");
const plans = require("../results/plans.json");

const p = E.interpret("A sombre fugue in D minor for three voices", 0);
assert.strictEqual(p.keyName, "D minor"); assert.strictEqual(p.voices, 3); assert.ok(p.tempo <= 60);
assert.strictEqual(E.interpret("joyful, B flat major, strings", 0).keyName, "B♭ major");
assert.strictEqual(E.interpret("in F sharp minor, a duet", 0).keyName, "F♯ minor");
assert.strictEqual(E.interpret("a quick little fugue in a minor key", 0).mode, "minor");
assert.strictEqual(E.interpret("five-part organ fugue", 0).timbre, "organ");

const CONS = new Set([0, 3, 4, 7, 8, 9]);
for (const q of ["A sombre fugue in D minor for three voices", "Bright and lively, four voices, in G major", "a short duet in E minor", "Slow, chromatic, five voices"]) {
  const g = E.generate(q, plans, 1);
  assert.ok(g.entries.length >= g.voices, q + ": every voice states the subject");
  const midi = E.toMidi(g); assert.ok(midi.length > 100);
  // reproducible from request + seed
  assert.deepStrictEqual(E.generate(q, plans, 1).notes, g.notes, q + ": deterministic");
  // strong-beat dissonance stays rare
  const at = (v, t) => { for (const n of g.notes) if (n.v === v && n.on <= t && t < n.on + n.d) return n.m; return null; };
  let diss = 0, pairs = 0;
  for (let t = 0; t < g.length; t += 1) {
    const ps = [...Array(g.voices).keys()].map((v) => at(v, t)).filter((x) => x !== null), bass = Math.min(...ps);
    for (let i = 0; i < ps.length; i++) for (let j = i + 1; j < ps.length; j++) {
      const ic = Math.abs(ps[i] - ps[j]) % 12; pairs++;
      if (!CONS.has(ic) && !(ic === 5 && Math.min(ps[i], ps[j]) !== bass)) diss++;
    }
  }
  assert.ok(diss / pairs < 0.08, `${q}: strong-beat dissonance ${(100 * diss / pairs).toFixed(1)}%`);
}
console.log("engine tests passed");
