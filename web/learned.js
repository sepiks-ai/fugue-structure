/* Learned fugue writing: the counterpoint model (trained on the 48 fugues of Kyle Rother's open edition
   of the Well-Tempered Clavier, CC BY 4.0) and the learned plan model.
   The feature code mirrors src/fugue/learn/grid.py exactly; tests/test_learned.js checks the two agree. */
(function (root) {
  "use strict";
  const STEP = 0.25, OWN = 12, OFFS = [-4, -3, -2, -1, 0, 1, 2, 4], IV = 16;
  const HOLD = 2 * IV + 1, REST = 2 * IV + 2, N_OUT = 2 * IV + 3;
  const SLOTS = ["above1", "above2", "below1", "below2"];
  const mod = (a, n) => ((a % n) + n) % n;
  const clampI = (x, lim) => Math.max(-lim, Math.min(lim, x));

  // ---------------------------------------------------------------- feature tables (same order as grid.py)
  const TABLES = [];
  for (let k = 1; k <= OWN; k++) TABLES.push([`own${k}`, 2 + 2 * (2 * IV + 1)]);
  TABLES.push(["degree", 24], ["register", 10], ["pos", 32], ["weight", 5], ["role", 15]);
  for (const s of SLOTS) for (const o of OFFS) { const k = `${s}${o >= 0 ? "+" : ""}${o}`; TABLES.push([k, 27], [k + "oct", 7]); }
  const OFFSET = []; { let a = 0; for (const [, n] of TABLES) { OFFSET.push(a); a += n; } }
  const IDX = Object.fromEntries(TABLES.map(([n], i) => [n, i]));

  function weightOf(pos, bar, num) {
    if (pos === 0) return 4;
    const beat = [6, 9, 12].includes(num) ? Math.floor(bar / Math.floor(num / 3)) : Math.floor(bar / num);
    if (bar % 2 === 0 && pos === bar / 2 && [4, 2, 12].includes(num)) return 3;
    if (beat && pos % beat === 0) return 2;
    if (pos % 2 === 0) return 1;
    return 0;
  }
  const ownCat = (p, onset, p0) => (p < 0 ? 1 : 2 + (onset ? 0 : 2 * IV + 1) + clampI(p - p0, IV) + IV);
  function otherCat(p, onset, p0) {           // known note or rest
    if (p < 0) return [2, 0];
    const d = p - p0;
    return [3 + (onset ? 0 : 12) + mod(d, 12), 1 + Math.max(-3, Math.min(2, Math.floor(d / 12))) + 3];
  }
  // grid: {pitch: Int16Array[], onset: Uint8Array[], known: Uint8Array[] (1 = may be seen)} per voice, top voice first
  function features(G, v, t, tonic, mode, bar, num, out) {
    const P = G.pitch[v], O = G.onset[v];
    let p0 = null;
    for (let s = t - 1; s >= 0; s--) if (P[s] >= 0) { p0 = P[s]; break; }
    if (p0 === null) return null;
    let f = 0;
    for (let k = 1; k <= OWN; k++) { const s = t - k; out[f] = OFFSET[f] + (s < 0 ? 0 : ownCat(P[s], O[s], p0)); f++; }
    out[f] = OFFSET[f] + mod(p0 - tonic, 12) + (mode === "minor" ? 12 : 0); f++;
    out[f] = OFFSET[f] + Math.max(0, Math.min(9, Math.floor((p0 - (60 + tonic)) / 6) + 5)); f++;
    const pos = t % bar; out[f] = OFFSET[f] + Math.min(31, pos); f++;
    out[f] = OFFSET[f] + weightOf(pos, bar, num); f++;
    const nv = G.pitch.length; out[f] = OFFSET[f] + (nv >= 2 ? Math.min(14, [0, 0, 1, 3, 6, 10][Math.min(nv, 5)] + v) : 0); f++;
    const above = [], below = [];
    for (let u = v - 1; u >= 0 && above.length < 2; u--) above.push(u);
    for (let u = v + 1; u < nv && below.length < 2; u++) below.push(u);
    for (const lst of [above, below]) for (let i = 0; i < 2; i++) {
      const u = i < lst.length ? lst[i] : null;
      for (const o of OFFS) {
        const s = t + o;
        let a = 0, b = 0;
        if (u !== null && s >= 0 && s < G.pitch[u].length) {
          if (G.known[u][s]) [a, b] = otherCat(G.pitch[u][s], G.onset[u][s], p0); else [a, b] = [1, 0];
        }
        out[f] = OFFSET[f] + a; f++; out[f] = OFFSET[f] + b; f++;
      }
    }
    return p0;
  }

  // ---------------------------------------------------------------- the network
  let NET = null;
  function loadNet(meta, buf) {                 // meta: {hidden, rows, scales offset...}; buf: ArrayBuffer of int8 + float32 parts
    const H = meta.hidden, R = meta.rows, dv = new DataView(buf);
    let o = 0;
    const emb = new Float32Array(R * H);
    const q = new Int8Array(buf, 0, R * H); o = R * H; o = Math.ceil(o / 4) * 4;
    const sc = new Float32Array(buf, o, R); o += R * 4;
    for (let r = 0; r < R; r++) for (let h = 0; h < H; h++) emb[r * H + h] = q[r * H + h] * sc[r];
    const take = (n) => { const a = new Float32Array(buf.slice(o, o + n * 4)); o += n * 4; return a; };
    const b0 = take(H), W1 = take(H * H), b1 = take(H), Wo = take(N_OUT * H), bo = take(N_OUT);
    NET = { H, emb, b0, W1, b1, Wo, bo, h0: new Float32Array(H), h1: new Float32Array(H), logits: new Float32Array(N_OUT), x: new Int32Array(TABLES.length) };
    void dv;
    return NET;
  }
  function forward(idx) {                       // idx: row indices (already offset); returns logits
    const { H, emb, b0, W1, b1, Wo, bo, h0, h1, logits } = NET;
    h0.set(b0);
    for (let f = 0; f < idx.length; f++) { const r = idx[f] * H; for (let h = 0; h < H; h++) h0[h] += emb[r + h]; }
    for (let h = 0; h < H; h++) if (h0[h] < 0) h0[h] = 0;
    for (let i = 0; i < H; i++) { let s = b1[i]; const w = i * H; for (let h = 0; h < H; h++) s += W1[w + h] * h0[h]; h1[i] = s > 0 ? s : 0; }
    for (let k = 0; k < N_OUT; k++) { let s = bo[k]; const w = k * H; for (let h = 0; h < H; h++) s += Wo[w + h] * h1[h]; logits[k] = s; }
    return logits;
  }

  // ---------------------------------------------------------------- writing the free voices
  // placed: [{voice, notes:[{on,d,m}], kind}] fixed material; voices sound only from their first entry.
  // Particles: several candidate continuations sample the model step by step (lowest voice first at each
  // step, so upper voices hear it); each bar the weaker half is replaced by copies of the stronger.
  function compose(opts) {
    const { voices: nv, placed, total, tonic, mode, ranges, seed = 1, particles = 8, temperature = 1.0, nucleus = 0.98, sweeps = 1, onProgress } = opts;
    const bar = 16, num = 4, T = Math.round(total / STEP);
    let r = seed >>> 0;
    const rnd = () => { r += 0x6d2b79f5; let x = r; x = Math.imul(x ^ (x >>> 15), x | 1); x ^= x + Math.imul(x ^ (x >>> 7), x | 61); return ((x ^ (x >>> 14)) >>> 0) / 4294967296; };
    // the fixed grid
    const fixedP = [], fixedO = [], isFixed = [], entered = [];
    for (let v = 0; v < nv; v++) { fixedP.push(new Int16Array(T).fill(-1)); fixedO.push(new Uint8Array(T)); isFixed.push(new Uint8Array(T)); entered.push(T); }
    for (const q of placed) {
      for (const x of q.notes) {
        const a = Math.round(x.on / STEP), b = Math.max(a + 1, Math.round((x.on + x.d) / STEP));
        for (let s = a; s < Math.min(b, T); s++) { fixedP[q.voice][s] = x.m; isFixed[q.voice][s] = 1; fixedO[q.voice][s] = s === a ? 1 : 0; }
      }
      entered[q.voice] = Math.min(entered[q.voice], Math.round(q.onset / STEP));
    }
    // a fixed passage that ends is followed by free music; a rest in fixed material stays free too
    const mkState = () => ({ pitch: fixedP.map((a) => a.slice()), onset: fixedO.map((a) => a.slice()), known: isFixed.map((a) => a.slice()), lp: 0, n: 0 });
    const score = (s) => s.lp / Math.max(1, s.n);            // average quality of each pitch choice, so more notes are not penalised
    let parts = [];
    for (let k = 0; k < particles; k++) parts.push(mkState());
    const order = [...Array(nv).keys()].reverse();         // bottom voice first
    const idx = new Int32Array(TABLES.length), probs = new Float32Array(N_OUT);
    const sample = (st, v, t, count) => {
      const p0 = features(st, v, t, tonic, mode, bar, num, idx);
      if (p0 === null) { st.pitch[v][t] = -1; st.known[v][t] = 1; return; }
      const lg = forward(idx);
      // allowed tokens: stay within the voice's range; no hold after a rest
      const [lo, hi] = ranges[v];
      let mx = -1e9;
      // texture limits (the only rules besides range and the final chord): not too many voices resting at
      // once, no rest longer than 6 beats unless an entry is coming, no note held longer than 8 beats
      let resting = 0; for (let u = 0; u < nv; u++) if (u !== v && t >= entered[u] && st.known[u][t] && st.pitch[u][t] < 0) resting++;
      let restRun = 0; for (let s = t - 1; s >= 0 && st.pitch[v][s] < 0 && restRun < 64; s--) restRun++;
      let holdRun = 0; for (let s = t - 1; s > 0 && !st.onset[v][s] && st.pitch[v][s] >= 0 && holdRun < 64; s--) holdRun++;
      let entrySoon = false; for (let s = t + 1; s < Math.min(T, t + 33); s++) if (isFixed[v][s]) { entrySoon = true; break; }
      const noRest = resting >= (nv >= 4 ? 2 : 1) || (restRun >= 24 && !entrySoon);
      for (let k = 0; k < N_OUT; k++) {
        let ok = true;
        if (k < HOLD) { const m = p0 + k - IV; if (m < lo || m > hi) ok = false; }
        else if (k === HOLD && (t === 0 || st.pitch[v][t - 1] < 0 || holdRun >= 31)) ok = false;
        else if (k === REST && noRest) ok = false;
        probs[k] = ok ? lg[k] / temperature : -1e9; if (probs[k] > mx) mx = probs[k];
      }
      let sum = 0; for (let k = 0; k < N_OUT; k++) { probs[k] = Math.exp(probs[k] - mx); sum += probs[k]; }
      // nucleus: sample from the smallest set of tokens holding most of the probability
      const sorted = [...probs.keys()].sort((x, y) => probs[y] - probs[x]);
      let acc = 0, cut = 0; for (; cut < N_OUT; cut++) { acc += probs[sorted[cut]] / sum; if (acc >= nucleus) { cut++; break; } }
      let u = rnd() * acc * sum, pick = sorted[0];
      for (let j = 0; j < cut; j++) { u -= probs[sorted[j]]; if (u <= 0) { pick = sorted[j]; break; } }
      // candidates are compared on their pitch choices only: the rhythm is left to the model, since
      // ranking by the full likelihood favours holding notes (the safest prediction) and slows everything down
      if (count && pick < HOLD) { let on = 0; for (let k = 0; k < HOLD; k++) on += probs[k]; st.lp += Math.log(probs[pick] / on + 1e-12); st.n++; }
      if (pick === REST) { st.pitch[v][t] = -1; st.onset[v][t] = 0; }
      else if (pick === HOLD) { st.pitch[v][t] = st.pitch[v][t - 1]; st.onset[v][t] = 0; }
      else { st.pitch[v][t] = p0 + pick - IV; st.onset[v][t] = 1; }
      st.known[v][t] = 1;
    };
    for (let t = 0; t < T; t++) {
      for (const st of parts) {
        const free = [];
        for (const v of order) {
          if (t < entered[v]) { st.known[v][t] = 1; continue; }     // silent before its first entry
          if (!isFixed[v][t]) free.push(v);
        }
        // first pass bottom-up, then each voice reconsiders knowing what all the others chose (a Gibbs sweep)
        for (const v of free) sample(st, v, t, sweeps === 0);
        for (let k = 0; k < sweeps; k++) for (const v of free) sample(st, v, t, k === sweeps - 1);
      }
      if (t % bar === bar - 1 && parts.length > 1) {           // keep the stronger half, duplicated
        parts.sort((a, b) => score(b) - score(a));
        const keep = parts.slice(0, Math.max(1, parts.length >> 1));
        parts = keep.concat(keep.map((s) => ({ pitch: s.pitch.map((a) => a.slice()), onset: s.onset.map((a) => a.slice()), known: s.known.map((a) => a.slice()), lp: s.lp, n: s.n })));
      }
      if (onProgress && t % 32 === 0) onProgress(t / T);
    }
    parts.sort((a, b) => score(b) - score(a));
    const best = parts[0], notes = [];
    for (let v = 0; v < nv; v++) {
      let t = 0;
      while (t < T) {
        if (isFixed[v][t] || best.pitch[v][t] < 0 || !best.onset[v][t] && !(t > 0 && isFixed[v][t - 1])) { t++; continue; }
        const m = best.pitch[v][t]; let u = t + 1;
        while (u < T && !isFixed[v][u] && best.pitch[v][u] === m && !best.onset[v][u]) u++;
        notes.push({ v, on: t * STEP, d: (u - t) * STEP, m, entry: -1, kind: "free" });
        t = u;
      }
    }
    return { notes, logp: best.lp };
  }

  // ---------------------------------------------------------------- plans from the learned plan model
  function samplePlan(PM, nv, rnd, opts = {}) {
    const pickW = (obj) => { const ks = Object.keys(obj); let u = rnd() * ks.reduce((a, k) => a + obj[k], 0); for (const k of ks) { u -= obj[k]; if (u <= 0) return k; } return ks[ks.length - 1]; };
    const orders = PM.exposition_order[String(nv)] || { [[...Array(nv).keys()].join(",")]: 1 };
    const full = Object.fromEntries(Object.entries(orders).filter(([k]) => k.split(",").length === nv));
    const order = pickW(Object.keys(full).length ? full : orders).split(",").map(Number);
    const lens = PM.length[String(nv)] || PM.length["3"];
    let L = lens[Math.floor(rnd() * lens.length)];
    if (opts.length === "short") L = Math.min(L, Math.max(nv + 4, 9)); else if (opts.length === "long") L = Math.max(L, 24);
    L = Math.max(nv + 3, Math.min(36, L));
    const entries = [];
    let t = 0;
    order.forEach((rank, k) => { entries.push({ rank, onset: t, degree: k % 2 ? 4 : 0, form: "plain" }); t += PM.exposition_gap; });
    let prev = entries[entries.length - 1], guard = 0;
    for (;;) {
      const pos = Math.min(3, Math.floor(4 * t / L));
      if (t >= L - 1.5 || (entries.length > nv + 2 && pickW(PM.end[String(pos)] || { go: 1, end: 0 }) === "end" && t > L * 0.6)) break;
      if (++guard > 60) break;
      const degree = +pickW(PM.degree[`${prev.degree}|${pos}`] || PM.degree_prior);
      let gap = Math.min(3, PM.gap_values[+pickW(PM.gap[String(pos)])]);
      if (opts.stretto) gap = Math.min(gap, 0.75 + 0.25 * rnd()); else gap = Math.max(gap, 0.5);
      let form = pickW(PM.form[String(pos)]);
      if (opts.inversion && form === "plain" && rnd() < 0.35) form = "inv";
      const move = pickW(PM.move);
      let rank = prev.rank + (move === "up" ? -1 : move === "down" ? 1 : 0);
      if (rank < 0 || rank >= nv) rank = Math.floor(rnd() * nv);
      t += gap;
      prev = { rank, onset: t, degree, form };
      entries.push(prev);
    }
    // the last entry comes home: the tonic, in the bass, if there is room after the last
    entries.push({ rank: nv - 1, onset: Math.min(L, t + 1.0), degree: 0, form: "plain" });
    return { order, entries, length: L + 1.5 };
  }

  const API = { features, forward, loadNet, compose, samplePlan, TABLES, OFFSET, IDX, HOLD, REST, N_OUT, STEP };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  else root.FugueLearned = API;
})(typeof self !== "undefined" ? self : this);
