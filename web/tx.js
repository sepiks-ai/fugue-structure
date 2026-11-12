/* The token composer in the browser: the transformer of src/fugue/tx/model.py (weights from
   scripts/export_tx.py) and the three learned layers of scripts/compose_hier.py: a plan learned from the 48,
   a subject written by the model and chosen for its likeness to Bach's, and the notes, with each planned entry
   written into its voice and the model writing everything else. Runs in a worker (tx-worker.js) or in node. */
(function (root) {
  "use strict";

  // ------------------------------------------------------------------ the model
  let M = null;                                     // {cfg, vocab, ID, W: {name: Float32Array}, ...}
  function load(meta, buf) {
    const W = {};
    for (const [name, t] of Object.entries(meta.tensors)) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      if (!t.int8) { W[name] = new Float32Array(buf, t.data, n); continue; }
      const rows = t.shape[0], cols = t.shape[1], sc = new Float32Array(buf, t.scale, rows), q = new Int8Array(buf, t.data, n);
      const out = new Float32Array(n);
      for (let r = 0; r < rows; r++) { const s = sc[r]; for (let c = 0; c < cols; c++) out[r * cols + c] = q[r * cols + c] * s; }
      W[name] = out;
    }
    const ID = {}; meta.vocab.forEach((t, i) => (ID[t] = i));
    M = { ...meta, ID, W, d: meta.cfg.d, L: meta.cfg.layers, H: meta.cfg.heads, hd: meta.cfg.d / meta.cfg.heads, V: meta.vocab.length };
    M.inv = new Float32Array(M.hd / 2); for (let i = 0; i < M.hd / 2; i++) M.inv[i] = 1 / Math.pow(10000, (2 * i) / M.hd);
    return M;
  }
  function erf(x) {                                 // Abramowitz and Stegun 7.1.26
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
  }
  function layerNorm(x, w, b, out) {
    let m = 0; for (let i = 0; i < x.length; i++) m += x[i]; m /= x.length;
    let v = 0; for (let i = 0; i < x.length; i++) v += (x[i] - m) * (x[i] - m); v /= x.length;
    const r = 1 / Math.sqrt(v + 1e-5);
    for (let i = 0; i < x.length; i++) out[i] = (x[i] - m) * r * w[i] + b[i];
    return out;
  }
  function matvec(W, x, rows, cols, out, bias) {   // out = W x (+ bias); W row-major [rows][cols]; cols % 4 == 0
    for (let r = 0; r < rows; r++) {
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0; const o = r * cols;
      for (let c = 0; c < cols; c += 4) { s0 += W[o + c] * x[c]; s1 += W[o + c + 1] * x[c + 1]; s2 += W[o + c + 2] * x[c + 2]; s3 += W[o + c + 3] * x[c + 3]; }
      out[r] = (bias ? bias[r] : 0) + s0 + s1 + s2 + s3;
    }
    return out;
  }
  // a cache of keys and values, so each new token costs one pass of the network. The piece's header stays at the
  // front; the rest is a sliding window of the last `cap` tokens. Positions are rotary, so attention sees only
  // distances: as the window slides, the header's keys are turned forward to stay just before it, and once the
  // first token falls out a CONT marker joins the header (as in training, where a window cut mid-piece has one).
  function Cache(cap) {
    const c = { cap, pos: 0, start: 0, n: 0, H0: 0, cont: false, hK: [], hV: [], K: [], V: [] };
    for (let l = 0; l < M.L; l++) { c.hK.push(new Float32Array(8 * M.d)); c.hV.push(new Float32Array(8 * M.d)); c.K.push(new Float32Array(cap * M.d)); c.V.push(new Float32Array(cap * M.d)); }
    return c;
  }
  const buf = {};
  function scratch(name, n) { if (!buf[name] || buf[name].length !== n) buf[name] = new Float32Array(n); return buf[name]; }
  function turnHead(c) {                           // move the header's keys one position later
    const { d, H, hd } = M;
    for (let i = 0; i < hd / 2; i++) {
      const cs = Math.cos(M.inv[i]), sn = Math.sin(M.inv[i]);
      for (let l = 0; l < M.L; l++) for (let t = 0; t < c.H0; t++) for (let h = 0; h < H; h++) {
        const o = t * d + h * hd + 2 * i, K = c.hK[l], a = K[o], b = K[o + 1];
        K[o] = a * cs - b * sn; K[o + 1] = a * sn + b * cs;
      }
    }
  }
  // push one token (with its absolute pitch, or 0) and return the logits for the next one (if wanted)
  function step(c, tok, aux, wantLogits, toHead) {
    if (!toHead && c.n === c.cap) {               // the oldest token leaves the window
      if (!c.cont) { c.cont = true; const p = c.pos; c.pos = c.start; forward(c, M.ID.CONT, 0, false, true); c.pos = p; }
      else turnHead(c);
      c.start++; c.n--;
    }
    return forward(c, tok, aux, wantLogits, toHead);
  }
  function forward(c, tok, aux, wantLogits, toHead) {
    const { d, H, hd, W } = M, pos = c.pos;
    const x = scratch("x", d), h = scratch("h", d), qkv = scratch("qkv", 3 * d), att = scratch("att", d), o = scratch("o", d);
    const ff = scratch("ff", 4 * d), sc = scratch("sc", c.cap + 9);
    for (let i = 0; i < d; i++) x[i] = W["emb.weight"][tok * d + i] + (W["pitch_emb.weight"] ? W["pitch_emb.weight"][(aux || 0) * d + i] : 0);
    const cos = scratch("cos", hd / 2), sin = scratch("sin", hd / 2);
    for (let i = 0; i < hd / 2; i++) { const a = pos * M.inv[i]; cos[i] = Math.cos(a); sin[i] = Math.sin(a); }
    const slot = toHead ? c.H0 : (c.start + c.n) % c.cap;
    for (let l = 0; l < M.L; l++) {
      const p = `blocks.${l}.`;
      layerNorm(x, W[p + "n1.weight"], W[p + "n1.bias"], h);
      matvec(W[p + "qkv.weight"], h, 3 * d, d, qkv);
      for (let part = 0; part < 2; part++) for (let hh = 0; hh < H; hh++) {
        const base = part * d + hh * hd;
        for (let i = 0; i < hd / 2; i++) {
          const a = qkv[base + 2 * i], b = qkv[base + 2 * i + 1];
          qkv[base + 2 * i] = a * cos[i] - b * sin[i]; qkv[base + 2 * i + 1] = a * sin[i] + b * cos[i];
        }
      }
      const K = toHead ? c.hK[l] : c.K[l], Vv = toHead ? c.hV[l] : c.V[l];
      for (let i = 0; i < d; i++) { K[slot * d + i] = qkv[d + i]; Vv[slot * d + i] = qkv[2 * d + i]; }
      const hk = c.hK[l], hv = c.hV[l], rk = c.K[l], rv = c.V[l];
      const nh = toHead ? c.H0 + 1 : c.H0, nr = toHead ? 0 : c.n + 1, scale = 1 / Math.sqrt(hd);
      for (let hh = 0; hh < H; hh++) {
        const qo = hh * hd; let mx = -Infinity, k = 0;
        for (let t = 0; t < nh; t++, k++) { let s = 0; const ko = t * d + qo; for (let i = 0; i < hd; i++) s += qkv[qo + i] * hk[ko + i]; sc[k] = s * scale; if (sc[k] > mx) mx = sc[k]; }
        for (let r = 0; r < nr; r++, k++) { const t = (c.start + r) % c.cap; let s = 0; const ko = t * d + qo; for (let i = 0; i < hd; i++) s += qkv[qo + i] * rk[ko + i]; sc[k] = s * scale; if (sc[k] > mx) mx = sc[k]; }
        let z = 0; for (let j = 0; j < k; j++) { sc[j] = Math.exp(sc[j] - mx); z += sc[j]; }
        for (let i = 0; i < hd; i++) att[qo + i] = 0;
        k = 0;
        for (let t = 0; t < nh; t++, k++) { const w = sc[k] / z, vo = t * d + qo; for (let i = 0; i < hd; i++) att[qo + i] += w * hv[vo + i]; }
        for (let r = 0; r < nr; r++, k++) { const t = (c.start + r) % c.cap, w = sc[k] / z, vo = t * d + qo; for (let i = 0; i < hd; i++) att[qo + i] += w * rv[vo + i]; }
      }
      matvec(W[p + "o.weight"], att, d, d, o);
      for (let i = 0; i < d; i++) x[i] += o[i];
      layerNorm(x, W[p + "n2.weight"], W[p + "n2.bias"], h);
      matvec(W[p + "ff.0.weight"], h, 4 * d, d, ff, W[p + "ff.0.bias"]);
      for (let i = 0; i < 4 * d; i++) ff[i] = 0.5 * ff[i] * (1 + erf(ff[i] / Math.SQRT2));
      matvec(W[p + "ff.2.weight"], ff, d, 4 * d, o, W[p + "ff.2.bias"]);
      for (let i = 0; i < d; i++) x[i] += o[i];
    }
    if (toHead) c.H0++; else c.n++;
    c.pos++; if (toHead) c.start = c.pos;
    if (!wantLogits) return null;
    layerNorm(x, W["norm.weight"], W["norm.bias"], h);
    return matvec(W["emb.weight"], h, M.V, d, new Float32Array(M.V));
  }

  // ------------------------------------------------------------------ tokens and the grammar
  const tok = (t) => M.ID[t];
  const name = (i) => M.vocab[i];
  const group = (i) => { const t = M.vocab[i], k = t.indexOf("_"); return k < 0 ? t : t.slice(0, k); };
  function durToken(steps) {
    const D = M.durs; let best = D[0];
    for (const x of D) if (Math.abs(x - steps) < Math.abs(best - steps)) best = x;
    return tok("DUR_" + best);
  }
  const BAR = 4 * 12;                               // 4/4, 12 steps per crotchet
  // the token grammar (scripts/sample_tx.py) plus the plan (scripts/compose_hier.py): forced runs at set times,
  // voices held while they state the subject, no running past the next forced event
  function Grammar(nv) {
    const g = { nv, prev: "BAR", pos: -1, voice: null, last: {}, bar: -1, forced: [], queue: [], busy: [], solo: null, horizon: null, limit: null };
    g.now = () => g.bar * BAR + Math.max(0, g.pos);
    g.nextForced = () => g.forced.reduce((a, f) => (a === null || f[0] < a ? f[0] : a), null);
    g.base = () => {
      const p = g.prev, ok = new Set(), POS = (a, b) => { for (let i = a; i < b; i++) ok.add(tok("POS_" + i)); };
      if (p === "BAR") { POS(0, BAR); for (const i of M.groups.METER) ok.add(i); }
      else if (p === "METER") POS(0, BAR);
      else if (p === "POS" || p === "DUR") {
        for (let v = 0; v < nv; v++) if (g.voice === null || v <= g.voice) ok.add(tok("VOICE_" + v));
        ok.add(tok("ENTRY"));
        if (p === "DUR") { POS(g.pos + 1, BAR); ok.add(tok("BAR")); ok.add(tok("EOS")); }
      } else if (p === "ENTRY") for (const i of M.groups.ROLE) ok.add(i);
      else if (p === "ROLE") for (const i of M.groups.SHIFT) ok.add(i);
      else if (p === "SHIFT") for (let v = 0; v < nv; v++) ok.add(tok("VOICE_" + v));
      else if (p === "EVOICE") ok.add(tok("VOICE_" + g.voice));
      else if (p === "VOICE") {
        if (!(g.voice in g.last)) for (const i of M.groups.PITCH) ok.add(i);
        else {
          const m0 = g.last[g.voice];
          for (let k = -M.max_int; k <= M.max_int; k++) if (m0 + k >= M.lo && m0 + k <= M.hi) ok.add(tok("INT_" + k));
          for (const i of M.groups.PITCH) ok.add(i);
        }
      } else if (p === "PITCH" || p === "INT") for (const i of M.groups.DUR) ok.add(i);
      return ok;
    };
    g.allowed = () => {
      if (g.queue.length) return [g.queue[0]];
      let ok = g.base(); const t = g.now();
      ok.delete(tok("ENTRY"));
      const held = new Set(g.busy.filter(([a, b]) => a <= t && t < b).map((x) => x[2]));
      if (g.solo !== null) for (let u = 0; u < nv; u++) if (u !== g.solo) held.add(u);
      ok = new Set([...ok].filter((i) => !(group(i) === "VOICE" && g.prev !== "SHIFT" && held.has(+name(i).slice(6)))));
      const tf = g.nextForced();
      const lims = [tf, g.limit].filter((x) => x !== null), lim = lims.length ? Math.min(...lims) : null;
      if (lim !== null) {
        ok = new Set([...ok].filter((i) => !(group(i) === "POS" && g.bar * BAR + +name(i).slice(4) > lim)));
        if ((g.bar + 1) * BAR > lim) ok.delete(tok("BAR"));
      }
      if (tf !== null) {
        ok.delete(tok("EOS"));
        if ((g.prev === "POS" || g.prev === "DUR") && t === tf) {
          const f = g.forced.filter((f) => f[0] === tf).reduce((a, b) => (b[1] > a[1] ? b : a));
          ok = new Set([...ok].filter((i) => group(i) === "VOICE" && +name(i).slice(6) > f[1])); ok.add(f[2][0]);
        }
      }
      if (g.limit !== null || (g.horizon !== null && t < g.horizon)) ok.delete(tok("EOS"));
      if (g.horizon !== null && t >= g.horizon + BAR && (g.prev === "DUR" || g.prev === "BAR")) ok = new Set([tok("EOS")]);
      return [...ok];
    };
    g.step = (i) => {
      const tf = g.nextForced();
      if (!g.queue.length && tf !== null && g.now() === tf) {
        const k = g.forced.findIndex((f) => f[0] === tf && f[2][0] === i);
        if (k >= 0) { g.queue = g.forced[k][2].slice(1); g.forced.splice(k, 1); }
      } else if (g.queue.length) g.queue.shift();
      let gr = group(i); const t = name(i);
      if (gr === "BAR") { g.bar++; g.pos = -1; g.voice = null; }
      else if (gr === "POS") { g.pos = +t.slice(4); g.voice = null; }
      else if (gr === "VOICE") { g.voice = +t.slice(6); if (g.prev === "SHIFT") gr = "EVOICE"; }
      else if (gr === "PITCH") g.last[g.voice] = +t.slice(6);
      else if (gr === "INT") g.last[g.voice] += +t.slice(4);
      g.prev = gr;
      return gr === "PITCH" || gr === "INT" ? g.last[g.voice] : 0;
    };
    return g;
  }
  function decode(ids) {                           // tokens -> notes {v, on, d, m} in crotchets, as tokens.decode
    const notes = []; let bar = -1, pos = 0; const last = {};
    for (let i = 0; i < ids.length; i++) {
      const t = name(ids[i]);
      if (t === "BAR") { bar++; pos = 0; }
      else if (t.startsWith("POS_")) pos = +t.slice(4);
      else if (t === "ENTRY") i += 3;
      else if (t.startsWith("VOICE_") && i + 2 < ids.length) {
        const p = name(ids[i + 1]), dt = name(ids[i + 2]);
        if (!(p.startsWith("PITCH_") || p.startsWith("INT_")) || !dt.startsWith("DUR_")) continue;
        const v = +t.slice(6), m = p.startsWith("PITCH_") ? +p.slice(6) : Math.min(M.hi, Math.max(M.lo, (last[v] ?? 60) + +p.slice(4)));
        last[v] = m; notes.push({ v, on: (bar * BAR + pos) / 12, d: +dt.slice(4) / 12, m }); i += 2;
      }
    }
    return notes;
  }

  // ------------------------------------------------------------------ the plan (learned from the 48)
  function pick(rnd, d) {
    const ks = Object.keys(d); let u = rnd() * ks.reduce((a, k) => a + d[k], 0);
    for (const k of ks) { u -= d[k]; if (u <= 0) return k; }
    return ks[ks.length - 1];
  }
  function samplePlan(PM, nv, rnd, opts) {
    const orders = PM.exposition_order[String(nv)] || { [[...Array(nv).keys()].join(",")]: 1 };
    const full = Object.fromEntries(Object.entries(orders).filter(([k]) => k.split(",").length === nv));
    const order = pick(rnd, Object.keys(full).length ? full : orders).split(",").map(Number);
    const lens = PM.length[String(nv)] || PM.length["3"];
    let L = lens[Math.floor(rnd() * lens.length)];
    if (opts.length === "short") L = Math.min(L, nv + 4); else if (opts.length === "long") L = Math.max(L, 16);
    L = Math.max(nv + 3, Math.min(opts.length === "long" ? 24 : 18, L));
    const entries = []; let t = 0;
    order.forEach((r, k) => { entries.push({ rank: r, onset: t, degree: k % 2 ? 4 : 0, form: "plain" }); t += PM.exposition_gap; });
    let prev = entries[entries.length - 1];
    for (let guard = 0; guard < 60; guard++) {
      const pos = Math.min(3, Math.floor(4 * t / L));
      if (t >= L - 1.5 || (entries.length > nv + 2 && pick(rnd, PM.end[String(pos)] || { go: 1, end: 0 }) === "end" && t > 0.6 * L)) break;
      const degree = +pick(rnd, PM.degree[`${prev.degree}|${pos}`] || PM.degree_prior);
      const gap = Math.max(1, Math.min(3, PM.gap_values[+pick(rnd, PM.gap[String(pos)])]));
      let form = pick(rnd, PM.form[String(pos)]); if (form !== "inv") form = "plain";
      if (opts.inversion && form === "plain" && rnd() < 0.3) form = "inv";
      const move = pick(rnd, PM.move); let rank = prev.rank + (move === "up" ? -1 : move === "down" ? 1 : 0);
      if (rank < 0 || rank >= nv) rank = Math.floor(rnd() * nv);
      t += gap; prev = { rank, onset: t, degree, form }; entries.push(prev);
    }
    entries.push({ rank: nv - 1, onset: t + 1, degree: 0, form: "plain" });
    return { order, entries };
  }

  // ------------------------------------------------------------------ composing
  const MAJ = [0, 2, 4, 5, 7, 9, 11], MIN = [0, 2, 3, 5, 7, 8, 10];
  const SPANS = [[4, 6], [6, 9], [8, 11], [12, 13]];
  function mulberry(a) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function sampleFrom(logits, ok, temp, rnd) {
    let mx = -Infinity; for (const i of ok) if (logits[i] > mx) mx = logits[i];
    let z = 0; const p = ok.map((i) => { const e = Math.exp((logits[i] - mx) / temp); z += e; return e; });
    let lz = 0; for (const i of ok) lz += Math.exp(logits[i] - mx);
    let u = rnd() * z, k = 0; for (; k < ok.length - 1; k++) { u -= p[k]; if (u <= 0) break; }
    return { i: ok[k], lp: logits[ok[k]] - mx - Math.log(lz) };
  }
  // opts: {tonic (0-11), mode, voices, seed, temperature, subjects, subjectTemperature, motion, leap, length, inversion,
  //        window}; onEvent({stage, ...}) reports the plan, each subject candidate, the subject and the notes so far
  function compose(PM, opts, onEvent) {
    const nv = Math.max(2, Math.min(5, opts.voices || 3)), rnd = mulberry((opts.seed || 0) * 2654435761 + 7);
    const temp = opts.temperature || 1, scale = opts.mode === "minor" ? MIN : MAJ, WIN = opts.window || 1024;
    const tonic = ((opts.tonic % 12) + 12) % 12;
    const { order, entries: plan } = samplePlan(PM, nv, rnd, opts);
    let wsum = 0; for (const [, w] of SPANS) wsum += w;
    let u = rnd() * wsum, spanQ = SPANS[0][0]; for (const [s, w] of SPANS) { u -= w; if (u <= 0) { spanQ = s; break; } }
    const span = spanQ * 12;
    onEvent && onEvent({ stage: "plan", order, plan, span: spanQ, voices: nv });
    const head = [tok("BOS"), tok("FORM_FUGUE"), tok(`KEY_${tonic}_${opts.mode}`), tok("METER_4/4"), tok(`VOICES_${nv}`)];

    function State() {
      const s = { seq: [], aux: [], g: Grammar(nv), cache: Cache(WIN), lp: 0, n: 0, logits: null };
      for (const t of head) { s.seq.push(t); s.aux.push(0); s.logits = step(s.cache, t, 0, true, true); }
      return s;
    }
    function push(s, t, a) { s.seq.push(t); s.aux.push(a); return step(s.cache, t, a, true, false); }
    function run(s, until, T, report) {
      const g = s.g; let count = 0;
      while (s.seq.length < 15000) {
        if (!g.queue.length && g.prev === "DUR" && g.now() >= until && !g.forced.length) return;
        const ok = g.allowed(); let i, lp = 0;
        if (ok.length === 1) i = ok[0];
        else { ({ i, lp } = sampleFrom(s.logits, ok, T, rnd)); if (group(i) === "PITCH" || group(i) === "INT") { s.lp += lp; s.n++; } }
        const a = g.step(i); s.logits = push(s, i, a);
        if (i === tok("EOS")) return;
        if (report && ++count % 40 === 0) report(s);
      }
    }
    // the subject: the plan's first voice alone, several candidates, kept by confidence plus likeness to Bach's
    const v0 = order[0], SM = PM.subject, triad = { 0: "tonic", 3: "third", 4: "third", 7: "fifth" };
    const shiftT = { log_density: 0.45 * (opts.motion || 0), steps: -0.15 * (opts.leap || 0), leaps: 0.1 * (opts.leap || 0) };
    let best = null;
    for (let c = 0; c < (opts.subjects || 6); c++) {
      const s = State(); s.g.solo = v0; s.g.limit = span;
      for (const t of ["BAR", "POS_0", "ENTRY", "ROLE_x1", "SHIFT_0", "VOICE_" + v0]) { const a = s.g.step(tok(t)); s.logits = push(s, tok(t), a); }
      run(s, span, opts.subjectTemperature || 0.9);
      const sub = decode(s.seq).filter((n) => n.v === v0 && n.on * 12 < span).map((n) => ({ on: n.on, d: Math.min(n.d, span / 12 - n.on), m: n.m }));
      if (sub.length < 4) continue;
      const ms = sub.map((n) => n.m), iv = ms.slice(1).map((m, k) => m - ms[k]), sp = span / 12;
      const meas = { log_density: Math.log(sub.length / sp), range: Math.max(...ms) - Math.min(...ms),
        steps: iv.filter((x) => Math.abs(x) <= 2).length / iv.length, leaps: iv.filter((x) => Math.abs(x) >= 5).length / iv.length,
        lengths: new Set(sub.map((n) => n.d)).size, longest: Math.max(...sub.map((n) => n.d)) / sp, peak_at: ms.indexOf(Math.max(...ms)) / (ms.length - 1) };
      let like = 0;
      for (const [k, [mu, sd]] of Object.entries(SM.measures)) { const z = (meas[k] - mu - (shiftT[k] || 0)) / sd; like -= 0.3 * z * z; }
      like += Math.log(SM.ending[triad[(((ms[ms.length - 1] - tonic) % 12) + 12) % 12] || "other"]);
      const score = s.lp / Math.max(1, s.n) + like;
      onEvent && onEvent({ stage: "candidate", k: c, notes: sub, score, kept: !best || score > best.score });
      if (!best || score > best.score) best = { score, s, sub };
    }
    const s = best.s, g = s.g, subj = best.sub;
    g.solo = null; g.limit = null; g.busy.push([0, span, v0]);
    onEvent && onEvent({ stage: "subject", notes: subj, voice: v0 });
    // the plan's later entries, written into their voices
    const R = {}; for (const [r, lo, hi] of PM.ranges[String(nv)] || PM.ranges["3"]) R[r] = [lo, hi];
    const rel = subj.map((n) => n.m - subj[0].m), mid = (Math.min(...rel) + Math.max(...rel)) / 2;
    const entries = [{ voice: v0, onset: 0, end: span / 12, degree: 0, inv: false, shift: 0 }];
    for (const e of plan.slice(1)) {
      const t0 = Math.round(e.onset * span / 12) * 12, [lo, hi] = R[e.rank], semis = scale[e.degree], sign = e.form === "inv" ? -1 : 1;
      let k = 0, bd = 1e9; for (let kk = -3; kk <= 3; kk++) { const dd = Math.abs(semis + 12 * kk + sign * mid - (lo + hi) / 2); if (dd < bd) { bd = dd; k = kk; } }
      const shift = semis + 12 * k; let prev = null;
      subj.forEach((n, j) => {
        const p = Math.max(M.lo, Math.min(M.hi, subj[0].m + shift + sign * (n.m - subj[0].m)));
        const on = t0 + Math.round(n.on * 12), d = Math.round(n.d * 12);
        const toks = j === 0
          ? [tok("ENTRY"), tok(sign < 0 ? "ROLE_inv" : "ROLE_x1"), tok("SHIFT_" + (((shift % 12) + 12) % 12)), tok("VOICE_" + e.rank), tok("VOICE_" + e.rank), tok("PITCH_" + p), durToken(d)]
          : [tok("VOICE_" + e.rank), tok("INT_" + Math.max(-M.max_int, Math.min(M.max_int, p - prev))), durToken(d)];
        g.forced.push([on, e.rank, toks]); prev = p;
      });
      g.busy.push([t0, t0 + span, e.rank]);
      entries.push({ voice: e.rank, onset: t0 / 12, end: (t0 + span) / 12, degree: e.degree, inv: sign < 0, shift });
    }
    const end = Math.max(...entries.map((e) => e.end)) * 12;
    g.horizon = end;
    onEvent && onEvent({ stage: "entries", entries, length: end / 12 + 4 });
    run(s, end + 2 * BAR, temp, (st) => onEvent && onEvent({ stage: "notes", notes: decode(st.seq), bar: st.g.bar }));
    const notes = decode(s.seq);
    const inEntry = (n) => entries.findIndex((e) => n.v === e.voice && n.on >= e.onset - 1e-6 && n.on < e.end - 1e-6);
    for (const n of notes) { n.entry = inEntry(n); n.kind = n.entry >= 0 ? "subject" : "free"; }
    return { notes, entries, subject: subj, order, plan, span: spanQ, voices: nv, tokens: s.seq.length,
             length: Math.max(...notes.map((n) => n.on + n.d)), seq: s.seq };
  }
  function init(meta, buf) {
    load(meta, buf);
    M.groups = {};
    meta.vocab.forEach((t, i) => { const k = t.indexOf("_"), gname = k < 0 ? t : t.slice(0, k); (M.groups[gname] = M.groups[gname] || []).push(i); });
  }
  const API = { init, compose, Cache, step: (c, t, a, head) => step(c, t, a, true, !!head), decode, samplePlan, get model() { return M; } };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  else root.FugueTx = API;
})(typeof self !== "undefined" ? self : this);
