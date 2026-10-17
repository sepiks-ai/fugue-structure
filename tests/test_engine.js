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
  // strong-beat dissonance: allowed when treated (suspension or passing note), rare otherwise
  const byV = [...Array(g.voices).keys()].map((v) => g.notes.filter((n) => n.v === v).sort((a, b) => a.on - b.on));
  const at = (v, t) => { const ns = byV[v]; for (let i = 0; i < ns.length; i++) if (ns[i].on <= t && t < ns[i].on + ns[i].d) return [ns, i]; return null; };
  const treated = (ns, i, t) => {
    const n = ns[i], prev = ns[i - 1], next = ns[i + 1];
    const resolves = next && n.m - next.m >= 1 && n.m - next.m <= 2;
    if (n.on < t - 1e-9) return true;                                   // held over the beat (suspension or held against)
    if (prev && prev.m === n.m && resolves) return true;                 // repeated preparation
    return !!(prev && next && Math.abs(n.m - prev.m) <= 2 && Math.abs(next.m - n.m) <= 2 && n.m !== prev.m && n.m !== next.m);
  };
  let diss = 0, bad = 0, pairs = 0;
  for (let t = 0; t < g.length; t += 1) {
    const s = [...Array(g.voices).keys()].map((v) => at(v, t)).filter(Boolean);
    const bass = Math.min(...s.map(([ns, i]) => ns[i].m));
    for (let i = 0; i < s.length; i++) for (let j = i + 1; j < s.length; j++) {
      const a = s[i][0][s[i][1]].m, b = s[j][0][s[j][1]].m, ic = Math.abs(a - b) % 12; pairs++;
      if (CONS.has(ic) || (ic === 5 && Math.min(a, b) !== bass)) continue;
      diss++;
      if (!treated(...s[i], t) && !treated(...s[j], t)) bad++;
    }
  }
  assert.ok(diss / pairs < 0.12, `${q}: strong-beat dissonance ${(100 * diss / pairs).toFixed(1)}%`);
  assert.ok(bad / pairs < 0.02, `${q}: untreated dissonance ${(100 * bad / pairs).toFixed(1)}%`);
  // a countersubject is composed and comes back; episodes are built from the material
  assert.ok(g.countersubject && g.countersubject.length >= 4, q + ": countersubject");
  assert.ok(g.counts.countersubjects >= g.voices - 1, q + ": countersubject against each exposition answer");
  // no parallel fifths or octaves on the quaver grid
  let par = 0, prev = null;
  for (let k = 0; k < g.length * 2; k++) {
    const ps = [...Array(g.voices).keys()].map((v) => { const x = at(v, k / 2); return x ? x[0][x[1]].m : null; });
    if (prev) for (let i = 0; i < g.voices; i++) for (let j = i + 1; j < g.voices; j++) {
      const [a0, b0, a1, b1] = [prev[i], prev[j], ps[i], ps[j]];
      if ([a0, b0, a1, b1].includes(null) || a1 === a0 || (a1 - a0) * (b1 - b0) <= 0) continue;
      const i0 = Math.abs(a0 - b0) % 12, i1 = Math.abs(a1 - b1) % 12;
      if (i0 === i1 && (i1 === 0 || i1 === 7)) par++;
    }
    prev = ps;
  }
  assert.ok(par <= 1, `${q}: ${par} parallel fifths or octaves`);
}
console.log("engine tests passed");
