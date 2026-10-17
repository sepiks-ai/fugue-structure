/* Fugue engine: request interpreter, subject composer, plan realisation, harmony, counterpoint, MIDI.
 * Runs in the browser (main thread or worker) and in Node (for tests).
 * Port of src/fugue/skeleton.py with minor keys, local-key transposition and a subject composer.
 */
(function (root) {
  "use strict";

  const MAJOR = [0, 2, 4, 5, 7, 9, 11];
  const MINOR = [0, 2, 3, 5, 7, 8, 10];
  const PC = { c: 0, d: 2, e: 4, f: 5, g: 7, a: 9, b: 11 };
  const NAMES_SHARP = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"];
  const CONSONANT = new Set([0, 3, 4, 7, 8, 9]);
  const mod = (a, n) => ((a % n) + n) % n;

  // ------------------------------------------------------------------ randomness
  function mulberry(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function hash(str) {
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }
  function gauss(r) { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  function pick(r, arr) { return arr[Math.floor(r() * arr.length)]; }
  function weighted(r, items) {            // items: [value, weight]
    const tot = items.reduce((s, x) => s + Math.max(0, x[1]), 0);
    let u = r() * tot;
    for (const [v, w] of items) { u -= Math.max(0, w); if (u <= 0) return v; }
    return items[items.length - 1][0];
  }

  // ------------------------------------------------------------------ 1. interpret the request
  const WORDS = {
    minor: ["sad", "sombre", "somber", "melancholy", "melancholic", "mournful", "dark", "tragic", "grave", "lament",
      "lamenting", "anguished", "brooding", "gloomy", "wistful", "serious", "stormy", "haunting", "sorrowful",
      "bleak", "desolate", "tormented", "grief", "grieving", "elegiac", "doleful", "pensive"],
    major: ["happy", "joyful", "joyous", "bright", "cheerful", "sunny", "triumphant", "festive", "playful", "jubilant",
      "radiant", "glorious", "merry", "celebratory", "serene", "sunlit", "exuberant", "optimistic", "hopeful"],
    leapUp: ["angular", "bold", "leaping", "dramatic", "heroic", "jagged", "wide", "wild", "fierce", "proud",
      "assertive", "striking", "spiky", "daring"],
    leapDown: ["smooth", "flowing", "lyrical", "gentle", "singing", "stepwise", "calm", "tender", "peaceful",
      "serene", "simple", "graceful", "legato", "sweet", "quiet", "pastoral"],
    moveUp: ["running", "busy", "dance", "dancing", "lively", "sparkling", "rapid", "virtuosic", "perpetual",
      "scurrying", "gigue", "jig", "bubbling", "nimble", "agile", "playful", "excited", "restless"],
    moveDown: ["stately", "solemn", "noble", "majestic", "grand", "hymn", "hymnlike", "chorale", "dignified",
      "ceremonial", "slow-moving", "broad", "measured"],
    chromatic: ["chromatic", "anguished", "tortured", "twisted", "chromaticism", "sinuous", "mysterious", "uneasy",
      "strange", "eerie", "uncanny", "tormented", "searching"],
    stretto: ["stretto", "strettos", "strettoes", "dense", "crowded", "overlapping", "packed", "intense", "climactic"],
    inversion: ["inversion", "inverted", "mirror", "mirrored", "upside"],
    short: ["short", "brief", "miniature", "little", "small", "tiny", "compact"],
    long: ["long", "grand", "extended", "epic", "big", "large", "expansive"],
  };
  const TEMPO = [
    [["very slow", "largo", "lento", "grave"], 46],
    [["slow", "slowly", "adagio", "solemn", "mournful", "lament", "sombre", "somber", "sad", "melancholy", "grieving", "elegiac"], 56],
    [["andante", "walking", "moderate", "moderato", "calm", "gentle", "stately", "measured", "pastoral"], 66],
    [["allegro", "lively", "brisk", "quick", "fast", "energetic", "dance", "dancing", "playful", "jig", "gigue",
      "cheerful", "joyful", "running", "nimble", "agile"], 96],
    [["presto", "very fast", "racing", "furious", "frantic", "breakneck", "virtuosic", "wild"], 118],
  ];
  const TIMBRES = { organ: "organ", piano: "piano", harpsichord: "harpsichord", strings: "strings",
    "string quartet": "strings", violin: "strings", violins: "strings", clavichord: "harpsichord" };
  const ACC = { "#": 1, "♯": 1, sharp: 1, "-sharp": 1, b: -1, "♭": -1, flat: -1, "-flat": -1 };

  function interpret(text, rawSeed) {
    const t = (text || "").trim();
    const low = " " + t.toLowerCase().replace(/[,.;:!?()"]/g, " ") + " ";
    const has = (w) => new RegExp("[\\s-]" + w.replace(/[-]/g, "\\-") + "[\\s-]").test(low);
    const count = (list) => list.filter(has).length;
    const heard = [];
    const p = { voices: null, tonicPc: null, mode: null, tempo: null, length: "medium", leap: 0, motion: 0,
      chromatic: false, stretto: false, inversion: false, template: null, timbre: "harpsichord", subject: null };

    // explicit subject in note syntax, for the curious: subject: D4/1 A4/1 ...
    const sm = t.match(/subject\s*[:=]\s*((?:[A-Ga-g][#b♯♭]?-?\d\/[\d.]+\s*)+)/);
    if (sm) { p.subject = sm[1].trim(); heard.push("your subject"); }

    // key: "D minor", "F sharp major", "B♭ minor", "in G"
    let km = t.match(/\b([A-Ga-g])\s*(#|♯|♭|-?sharp|-?flat|b(?=\s+(?:major|minor)))?\s+(major|minor|maj|min)\b/i);
    if (km && km[1] === "a" && !km[2] && /\ba\s+(major|minor)\s+(key|mode)/i.test(t)) km = null; // "a minor key"
    if (km && km[1] === "a" && !km[2] && /\b(a)\s+(major|minor)\s+(work|piece|fugue|composer)/i.test(t)) km = null;
    if (km) {
      p.tonicPc = mod(PC[km[1].toLowerCase()] + (ACC[(km[2] || "").toLowerCase()] || 0), 12);
      p.mode = /^min/i.test(km[3]) ? "minor" : "major";
    } else {
      const im = t.match(/\bin\s+([A-G])\s*(#|♯|♭|b\b|-?sharp|-?flat)?(?![a-z])/) ||
        t.match(/\bin\s+([a-g])\s*(#|♯|♭|-?sharp|-?flat)(?![a-z])/);
      if (im) p.tonicPc = mod(PC[im[1].toLowerCase()] + (ACC[(im[2] || "").toLowerCase()] || 0), 12);
      if (/\bminor\b/i.test(t)) p.mode = "minor";
      else if (/\bmajor\b/i.test(t)) p.mode = "major";
    }
    if (!p.mode) {
      const mi = count(WORDS.minor), ma = count(WORDS.major);
      if (mi > ma) p.mode = "minor"; else if (ma > mi) p.mode = "major";
    }

    // voices
    const vm = low.match(/[\s-](two|three|four|five|six|seven|eight|2|3|4|5|6|7|8)[\s-]*(voices?|parts?|voiced|part)[\s-]/);
    const NUM = { two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8 };
    if (vm) p.voices = NUM[vm[1]] || parseInt(vm[1], 10);
    else if (has("duet") || has("duo")) p.voices = 2;
    else if (has("trio")) p.voices = 3;
    else if (has("quartet")) p.voices = 4;
    else if (has("quintet")) p.voices = 5;
    if (p.voices && p.voices > 5) { heard.push(`${p.voices} voices is beyond Bach's book: using 5`); p.voices = 5; }

    // tempo
    const bm = low.match(/(\d{2,3})\s*(bpm|beats)/);
    if (bm) p.tempo = Math.max(36, Math.min(160, parseInt(bm[1], 10)));
    else for (const [ws, bpm] of TEMPO) if (ws.some((w) => low.includes(" " + w + " "))) p.tempo = bpm;

    // character
    p.leap = Math.max(-1, Math.min(1, count(WORDS.leapUp) - count(WORDS.leapDown)));
    p.motion = Math.max(-1, Math.min(1, count(WORDS.moveUp) - count(WORDS.moveDown)));
    p.chromatic = count(WORDS.chromatic) > 0;
    p.stretto = count(WORDS.stretto) > 0;
    p.inversion = count(WORDS.inversion) > 0;
    if (count(WORDS.short) > count(WORDS.long)) p.length = "short";
    else if (count(WORDS.long) > count(WORDS.short)) p.length = "long";
    const tm = low.match(/(?:no\.?|number|fugue|like)\s*#?\s*(\d{1,2})[\s]/);
    if (tm && +tm[1] >= 1 && +tm[1] <= 24) p.template = +tm[1];
    for (const [w, v] of Object.entries(TIMBRES)) if (low.includes(" " + w)) p.timbre = v;

    // fill the gaps from the seed, so an empty request still gives variety
    const r = mulberry(hash(t.toLowerCase()) ^ (rawSeed || 0));
    const chosen = [];
    if (!p.mode) { p.mode = r() < 0.5 ? "minor" : "major"; chosen.push("mode"); }
    if (p.tonicPc === null) {
      p.tonicPc = pick(r, p.mode === "minor" ? [2, 4, 7, 9, 0, 5] : [0, 2, 7, 5, 9, 10]);
      chosen.push("key");
    }
    if (!p.voices) { p.voices = weighted(r, [[3, 5], [4, 4], [2, 1]]); chosen.push("voices"); }
    if (!p.tempo) p.tempo = p.motion > 0 ? 92 : p.motion < 0 ? 60 : 72;

    p.keyName = keyName(p.tonicPc, p.mode);
    const kc = chosen.includes("key") ? " (key chosen)" : chosen.includes("mode") ? " (mode chosen)" : "";
    heard.unshift(`${p.keyName}${kc}`,
      `${p.voices} voices${chosen.includes("voices") ? " (chosen)" : ""}`, `${p.tempo} beats a minute`);
    if (p.leap > 0) heard.push("angular subject"); else if (p.leap < 0) heard.push("smooth subject");
    if (p.motion > 0) heard.push("running rhythm"); else if (p.motion < 0) heard.push("broad rhythm");
    if (p.chromatic) heard.push("chromatic");
    if (p.stretto) heard.push("stretto");
    if (p.inversion) heard.push("inversion");
    if (p.length !== "medium") heard.push(p.length);
    if (p.timbre !== "harpsichord") heard.push(p.timbre);
    const moods = new Set();
    for (const list of [WORDS.minor, WORDS.major, WORDS.leapUp, WORDS.leapDown, WORDS.moveUp, WORDS.moveDown, WORDS.chromatic])
      for (const w of list) if (has(w)) moods.add(w);
    for (const [ws] of TEMPO) for (const w of ws) if (low.includes(" " + w + " ")) moods.add(w);
    if (moods.size) heard.push("from: " + [...moods].slice(0, 6).join(", "));
    p.heard = heard;
    return p;
  }

  function keyName(pc, mode) {
    const flats = mode === "minor" ? [3, 5, 10, 0, 7] : [5, 10, 3, 8, 1];
    const sharpNames = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
    const flatNames = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"];
    let n = (flats.includes(pc) ? flatNames : sharpNames)[pc];
    if (mode === "major" && pc === 6) n = "F♯";
    return n + " " + mode;
  }

  // ------------------------------------------------------------------ key arithmetic
  function homeKey(p) {
    let tonic = 60 + p.tonicPc;
    if (p.tonicPc > 6) tonic -= 12;                 // keep the subject around middle C
    return { tonic, mode: p.mode, scale: p.mode === "minor" ? MINOR : MAJOR };
  }
  const scaleOf = (mode) => (mode === "minor" ? MINOR : MAJOR);
  function degMidi(deg, alt, tonic, scale) {
    return tonic + 12 * Math.floor(deg / 7) + scale[mod(deg, 7)] + alt;
  }
  function triadQuality(scale, deg) {
    const a = scale[mod(deg, 7)] + 12 * Math.floor(deg / 7);
    const b = degMidi(deg + 2, 0, 0, scale) - a;
    return b === 4 ? "major" : "minor";
  }

  // ------------------------------------------------------------------ 2. compose a subject
  const CELLS_ACTIVE = [[[0.25, 0.25, 0.25, 0.25], 3], [[0.5, 0.25, 0.25], 3], [[0.25, 0.25, 0.5], 2], [[0.5, 0.5], 2], [[0.75, 0.25], 1], [[1], 0.5]];
  const CELLS_MIDDLE = [[[0.5, 0.5], 4], [[1], 2], [[0.75, 0.25], 1.5], [[0.5, 0.25, 0.25], 1.5], [[0.25, 0.25, 0.5], 1], [[0.25, 0.25, 0.25, 0.25], 1]];
  const CELLS_BROAD = [[[1], 4], [[0.5, 0.5], 2.5], [[0.75, 0.25], 1.5], [[1.5, 0.5], 1.5], [[2], 1]];

  function rhythm(r, p) {
    const beats = p.motion > 0 ? pick(r, [6, 7, 8]) : p.motion < 0 ? pick(r, [7, 8, 9]) : pick(r, [6, 7, 8, 8]);
    const pool = p.motion > 0 ? CELLS_ACTIVE : p.motion < 0 ? CELLS_BROAD : CELLS_MIDDLE;
    const out = [];
    // head: something to remember
    const head = weighted(r, p.motion < 0 ? [[[2], 2], [[1, 1], 2], [[1.5, 0.5], 2]] :
      p.motion > 0 ? [[[1], 2], [[0.5, 0.5], 2], [[0.75, 0.25], 1]] : [[[1], 2], [[1, 1], 1], [[1.5, 0.5], 2], [[0.5, 0.5], 1]]);
    let used = head.reduce((s, x) => s + x, 0);
    out.push(...head);
    while (used < beats - 1) {
      let cell = weighted(r, pool);
      const len = cell.reduce((s, x) => s + x, 0);
      if (used + len > beats - 1) cell = [beats - 1 - used];
      out.push(...cell);
      used += cell.reduce((s, x) => s + x, 0);
    }
    // close on a beat: a pair of quavers or a crotchet, then the final note
    out.push(...(r() < 0.5 ? [0.5, 0.5] : [1]));
    out.push(1);
    return out;
  }

  function melody(r, durs, p, key) {
    const n = durs.length, target = 0.82 - 0.22 * p.leap;
    const degs = [];
    let d = r() < 0.65 ? 0 : -3, lastLeap = 0, penult = null;
    degs.push(d);
    const climaxAt = Math.floor(n * (0.4 + 0.3 * r()));
    for (let i = 1; i < n; i++) {
      const remaining = n - 1 - i;
      let step;
      if (remaining === 1) {                             // penultimate: a step from the tonic
        const goal = 7 * Math.round(d / 7);
        penult = goal + (r() < 0.55 ? -1 : 1);
        step = penult - d;
      } else if (remaining === 0) {
        step = 7 * Math.round(d / 7) - d;               // final: the tonic
      } else if (lastLeap) {
        step = -Math.sign(lastLeap);                     // recover from a leap by step
      } else if (r() < target) {
        const toward = i < climaxAt ? 1 : -1;
        step = r() < 0.7 ? toward : -toward;
        if (r() < 0.12) step = 0;
      } else {
        const size = weighted(r, [[2, 4], [3, 3], [4, 2 + 2 * Math.max(0, p.leap)], [5, 0.5 + p.leap], [7, 0.3 + 0.6 * Math.max(0, p.leap)]]);
        step = (i < climaxAt ? 1 : -1) * size * (r() < 0.8 ? 1 : -1);
      }
      if (remaining > 1 && d + step > 9) step = -1;
      if (remaining > 1 && d + step < -3) step = 1;
      lastLeap = Math.abs(step) >= 2 ? step : 0;
      d += step;
      degs.push(d);
    }
    const alts = new Array(n).fill(0);
    const pitch = (i) => degMidi(degs[i], alts[i], key.tonic, key.scale);
    if (p.mode === "minor") {                           // leading tone below the tonic; raised sixth leading up to it
      for (let i = 0; i < n - 1; i++) {
        if (mod(degs[i], 7) === 6 && degs[i + 1] === degs[i] + 1) {
          alts[i] = 1;
          if (i > 0 && degs[i - 1] === degs[i] - 1) alts[i - 1] = 1;
        }
      }
    }
    if (p.chromatic) {
      for (let i = 1; i < n - 1; i++) {
        if (r() > 0.7) continue;
        const up = degs[i] - degs[i - 1] === 1 && degs[i + 1] - degs[i] === 1;
        const down = degs[i - 1] - degs[i] === 1 && degs[i] - degs[i + 1] === 1;
        if (degs[i - 1] === degs[i + 1] && Math.abs(degs[i] - degs[i - 1]) === 1 && durs[i] <= 0.5) {
          const target = pitch(i - 1) - 1;               // chromatic lower neighbour, a semitone below
          degs[i] = degs[i - 1] - 1;
          alts[i] = target - degMidi(degs[i], 0, key.tonic, key.scale);
        } else if (up && mod(degs[i], 7) === 3) {
          alts[i] = 1;                                   // sharpened fourth rising to the dominant
        } else if (down && mod(degs[i], 7) === 1) {
          alts[i] = -1;                                  // flattened second falling to the tonic
        } else if (down && mod(degs[i], 7) === 5 && p.mode === "major") {
          alts[i] = -1;                                  // borrowed flat sixth falling to the dominant
        }
      }
    }
    return { degs, alts };
  }

  function scoreSubject(sub, p, key) {
    const m = sub.notes.map((x) => x.m), n = m.length;
    let s = 0;
    const ints = m.slice(1).map((x, i) => x - m[i]);
    const steps = ints.filter((x) => Math.abs(x) <= 2).length / ints.length;
    s -= 4 * Math.abs(steps - (0.82 - 0.22 * p.leap));
    const hi = Math.max(...m), his = m.filter((x) => x === hi).length;
    if (his === 1) s += 2;
    const at = m.indexOf(hi) / n;
    if (at > 0.25 && at < 0.8) s += 1;
    for (const x of ints) {
      const a = Math.abs(x);
      if (a === 6) s -= 5;
      if (a === 3 && x !== 0) {
        const pair = [mod(m[ints.indexOf(x)] - key.tonic, 12), mod(m[ints.indexOf(x) + 1] - key.tonic, 12)];
        if (pair.includes(8) && pair.includes(11)) s -= 5;    // augmented second
      }
      if (a > 12) s -= 10;
      if (a === 10 || a === 11) s -= 3;
    }
    for (let i = 1; i < ints.length; i++) if (Math.abs(ints[i]) > 2 && Math.abs(ints[i - 1]) > 2 && Math.sign(ints[i]) === Math.sign(ints[i - 1])) s -= 2;
    let run = 1;
    for (let i = 1; i < n; i++) { run = m[i] === m[i - 1] ? run + 1 : 1; if (run > 2) s -= 2; }
    let turns = 0;
    for (let i = 1; i < ints.length; i++) if (Math.sign(ints[i]) && Math.sign(ints[i - 1]) && Math.sign(ints[i]) !== Math.sign(ints[i - 1])) turns++;
    if (turns >= 2 && turns <= n / 2) s += 1;
    if (p.leap >= 0 && ints.some((x) => Math.abs(x) >= 5)) s += 0.7;
    if (mod(m[n - 1] - key.tonic, 12) === 0) s += 1;
    const span = hi - Math.min(...m);
    if (span > 14) s -= 3;
    if (span < 7) s -= 2;
    if (span >= 8 && span <= 12) s += 1;
    for (let i = 1; i < n; i++) if (m[i] === m[i - 1]) s -= 1;                        // repeated notes
    for (let i = 3; i < n; i++) if (m[i] === m[i - 2] && m[i - 1] === m[i - 3]) s -= 3;  // x y x y oscillation
    const head = m.slice(0, 4);
    if (Math.abs(m[1] - m[0]) >= 3 || Math.max(...head) - Math.min(...head) >= 5) s += 1.5;  // a head with a profile
    if (m[1] < m[0] && Math.abs(m[1] - m[0]) <= 2 && sub.notes[0].d >= 1 && sub.notes[1].d >= 1) s -= 1.5;
    const ds = sub.notes.map((x) => x.d);
    if (ds.some((d) => d >= 1) && ds.some((d) => d <= 0.5)) s += 1;                      // rhythmic contrast
    if (n > 18) s -= (n - 18) * 0.7;
    if (n < 7) s -= 2;
    for (const x of sub.notes) {                                                          // long notes on stable degrees
      const pc = mod(x.m - key.tonic, 12);
      if (x.d >= 1 && x.t > 0 && ![0, key.scale[2], 7, key.scale[4], 2, 11].includes(pc)) s -= 0.8;
    }
    // beats on tonic or dominant harmony
    const I = [0, key.scale[2], 7], V = [7, 11, 2];
    for (const x of sub.notes) if (x.t % 1 === 0 && (I.includes(mod(x.m - key.tonic, 12)) || V.includes(mod(x.m - key.tonic, 12)))) s += 0.3;
    return s;
  }

  function composeSubject(p, key, r) {
    let best = null;
    for (let k = 0; k < 700; k++) {
      const durs = rhythm(r, p);
      const { degs, alts } = melody(r, durs, p, key);
      let t = 0;
      const notes = durs.map((d, i) => { const x = { t, d, deg: degs[i], alt: alts[i], m: degMidi(degs[i], alts[i], key.tonic, key.scale) }; t += d; return x; });
      const sub = { notes, span: t };
      const s = scoreSubject(sub, p, key) + 0.4 * gauss(r);
      if (!best || s > best.s) best = { s, sub };
    }
    return best.sub;
  }

  function parseSubject(text, key) {
    let t = 0;
    const notes = [];
    for (const tok of text.split(/\s+/)) {
      const m = tok.match(/^([A-Ga-g])([#b♯♭]?)(-?\d)\/([\d.]+)$/);
      if (!m) continue;
      const midi = 12 * (parseInt(m[3], 10) + 1) + PC[m[1].toLowerCase()] + ({ "#": 1, "♯": 1, b: -1, "♭": -1 }[m[2]] || 0);
      const d = parseFloat(m[4]);
      // nearest scale degree with an alteration
      const rel = midi - key.tonic, oct = Math.floor(rel / 12), pc = mod(rel, 12);
      let deg = 0;
      for (let i = 6; i >= 0; i--) if (key.scale[i] <= pc) { deg = i; break; }
      notes.push({ t, d, deg: deg + 7 * oct, alt: pc - key.scale[deg], m: midi });
      t += d;
    }
    return notes.length >= 3 ? { notes, span: t } : null;
  }

  // ------------------------------------------------------------------ 3. choose and realise a Bach plan
  function choosePlan(plans, p, r) {
    const usable = plans.filter((x) => x.usable);
    if (p.template) {
      const t = usable.find((x) => x.number === p.template);
      if (t) return t;
    }
    let best = [], bestS = -1e9;
    for (const x of usable) {
      let s = -3 * Math.abs(x.voices - p.voices);
      if (x.mode === p.mode) s += 1.5;
      if (p.stretto) s += 4 * x.n_overlaps / x.n_entries;
      if (p.inversion) s += 6 * x.n_inverted / x.n_entries;
      if (p.length === "long") s += x.n_entries / 12;
      s += 0.6 * r();
      if (s > bestS + 1e-9) { best = [x]; bestS = s; }
    }
    return best[0];
  }

  function placeEntry(sub, e, key) {
    const tr = e.transform;
    const sc = tr.includes("AUG") ? 2 : tr.includes("DIM") ? 0.5 : 1;
    const inv = tr.startsWith("INV");
    const ms = mod(e.midi_shift, 12);
    let localTonic, localMode;
    if (ms === 0 || ms === 5 || ms === 7) {
      localTonic = key.tonic + e.midi_shift;
      localMode = key.mode;
    } else {
      localTonic = degMidi(e.dia_shift, 0, key.tonic, key.scale);
      localMode = triadQuality(key.scale, e.dia_shift);
    }
    const ls = scaleOf(localMode), d0 = sub.notes[0].deg;
    const notes = sub.notes.map((x) => {
      const deg = inv ? 2 * d0 - x.deg : x.deg;
      let alt = inv ? 0 : x.alt;
      if (localMode !== key.mode && !inv && mod(x.deg, 7) === 6 && key.mode === "minor" && x.alt === 1) alt = 1;
      return { t: x.t * sc, d: x.d * sc, m: degMidi(deg, alt, localTonic, ls) };
    });
    return { notes, localKey: keyName(mod(localTonic, 12), localMode), inv, sc };
  }

  function strongClashes(a, b) {
    const at = (ns, t) => { for (const x of ns) if (x.on <= t && t < x.on + x.d) return x.m; return null; };
    const lo = Math.max(a[0].on, b[0].on), hi = Math.min(a[a.length - 1].on + a[a.length - 1].d, b[b.length - 1].on + b[b.length - 1].d);
    let n = 0;
    for (let t = Math.ceil(lo); t < hi; t += 1) {
      const x = at(a, t), y = at(b, t);
      if (x !== null && y !== null) { const ic = Math.abs(x - y) % 12; if (!CONSONANT.has(ic) && ic !== 5) n++; }
    }
    return n;
  }

  function parallels(a, b) {               // parallel fifths/octaves between two placed note lists
    const at = (ns, t) => { for (const x of ns) if (x.on <= t && t < x.on + x.d) return x.m; return null; };
    const lo = Math.max(a[0].on, b[0].on), hi = Math.min(a[a.length - 1].on + a[a.length - 1].d, b[b.length - 1].on + b[b.length - 1].d);
    let n = 0, pa = null, pb = null;
    for (let t = lo; t < hi; t += 0.25) {
      const x = at(a, t), y = at(b, t);
      if (x !== null && y !== null && pa !== null && pb !== null && x !== pa && (x - pa) * (y - pb) > 0) {
        const i0 = Math.abs(pa - pb) % 12, i1 = Math.abs(x - y) % 12;
        if (i0 === i1 && (i1 === 0 || i1 === 7)) n++;
      }
      pa = x; pb = y;
    }
    return n;
  }

  function realise(sub, plan, key, p) {
    const scale = sub.span / plan.subject_span;
    let ents = plan.entries.slice();
    const maxN = p.length === "short" ? Math.max(plan.voices + 3, 7) : p.length === "long" ? 99 : 16;
    ents = ents.slice(0, maxN);
    const t0 = ents[0].onset;
    const placed = [];
    const seen = new Set();
    for (const e of ents) {
      const first = !seen.has(e.voice);
      seen.add(e.voice);
      const base = Math.round((e.onset - t0) * scale);
      const pe = placeEntry(sub, e, key);
      let best = null;
      for (const off of [0, 1, -1, 2, -2, 3, -3, 4, 6, 8]) {
        const on = base + off;
        if (on < 0) continue;
        const notes = pe.notes.map((x) => ({ on: on + x.t, d: x.d, m: x.m }));
        const end = notes[notes.length - 1].on + notes[notes.length - 1].d;
        if (placed.some((q) => q.voice === e.voice && on < q.end && q.onset < end)) continue;
        let clash = 0, par = 0;
        for (const q of placed) if (on < q.end && q.onset < end) { clash += strongClashes(notes, q.notes); par += parallels(notes, q.notes); }
        clash += 2 * par;
        const s = clash + 0.3 * Math.abs(off);
        if (!best || s < best.s) best = { s, clash, on, notes, end };
      }
      if (!best || (best.clash > 4 && !first)) continue;
      placed.push({ voice: e.voice, onset: best.on, end: best.end, notes: best.notes, transform: e.transform,
        inv: pe.inv, localKey: pe.localKey, clash: best.clash });
    }
    return placed;
  }

  // ------------------------------------------------------------------ 4. harmony
  function chordSet(key) {
    const sc = key.scale, out = [];
    const triad = (deg) => [0, 2, 4].map((k) => mod(degMidi(deg + k, 0, 0, sc), 12));
    const roman = key.mode === "major" ? ["I", "ii", "iii", "IV", "V", "vi", "vii°"] : ["i", "ii°", "III", "iv", "v", "VI", "VII"];
    for (let d = 0; d < 7; d++) {
      let pcs = triad(d), name = roman[d];
      if (key.mode === "minor" && d === 4) { pcs = [7, 11, 2]; name = "V"; }
      out.push({ name, deg: d, root: pcs[0], pcs: new Set(pcs) });
    }
    out.push({ name: "V7", deg: 4, root: 7, pcs: new Set([7, 11, 2, 5]) });
    if (key.mode === "minor") out.push({ name: "vii°", deg: 6, root: 11, pcs: new Set([11, 2, 5]) });
    for (let tdeg = 1; tdeg <= 5; tdeg++) {
      const tr = triad(tdeg);
      if (mod(tr[2] - tr[0], 12) !== 7) continue;          // only to major or minor triads
      const root = mod(tr[0] + 7, 12);
      out.push({ name: "V/" + roman[tdeg], deg: null, target: tdeg, root, pcs: new Set([root, mod(root + 4, 12), mod(root + 7, 12)]) });
    }
    return out;
  }

  function chooseChords(fixedAt, nSlots, per, key) {
    const chords = chordSet(key);
    const tonicMajor = { name: key.mode === "minor" ? "I (Picardy)" : "I", deg: 0, root: 0, pcs: new Set([0, 4, 7]) };
    const v7 = chords.find((c) => c.name === "V7");
    const emit = (slot, c) => {
      let s = 0, lowest = null;
      for (let k = 0; k < per; k++) {
        const t = slot * per + k;
        if (t >= fixedAt.length) break;
        const w = k === 0 ? 1.5 : k % 2 === 0 ? 1 : 0.4;
        for (const m of fixedAt[t]) {
          const pc = mod(m - key.tonic, 12);
          s += c.pcs.has(pc) ? w : (k % 2 === 0 ? -1.5 * w : -0.3 * w);
          if (lowest === null || m < lowest) lowest = m;
        }
      }
      if (lowest !== null && mod(lowest - key.tonic, 12) === c.root) s += 0.5;
      if (c.target) s -= 0.3;
      return s;
    };
    const trans = (a, b) => {
      if (a.target) return b.deg === a.target ? 2 : -3;
      if (a === b) return -0.3;
      const iv = mod(b.root - a.root, 12);
      if (a.deg === 4 && b.deg === 3) return -1.5;
      return { 5: 1, 7: 0.2, 8: 0.5, 9: 0.5, 3: 0, 4: 0, 2: 0.3, 10: -0.2, 0: -0.3, 1: -1, 11: -1, 6: -1 }[iv];
    };
    const V = [];
    for (let i = 0; i < nSlots; i++) {
      const opts = i >= nSlots - 2 ? [tonicMajor] : i === nSlots - 3 ? [v7] : chords;
      const row = new Map();
      for (const c of opts) {
        const e = emit(i, c);
        if (i === 0) { row.set(c, { s: e + (c.deg === 0 ? 2 : 0), prev: null }); continue; }
        let bp = null, bs = -1e9;
        for (const [pc, v] of V[i - 1]) { const s = v.s + trans(pc, c); if (s > bs) { bs = s; bp = pc; } }
        row.set(c, { s: bs + e, prev: bp });
      }
      V.push(row);
    }
    let cur = null, bs = -1e9;
    for (const [c, v] of V[nSlots - 1]) if (v.s > bs) { bs = v.s; cur = c; }
    const seq = [cur];
    for (let i = nSlots - 1; i > 0; i--) { cur = V[i].get(cur).prev; seq.push(cur); }
    return seq.reverse();
  }

  // ------------------------------------------------------------------ 5. free counterpoint
  function fill(placed, plan, key, opts) {
    const { seed = 1, beam = 30, onProgress } = opts || {};
    const r = mulberry(seed ^ 0x9e3779b9);
    const nv = plan.voices, step = 0.5, spb = 2;
    const off = (opts && opts.rangeOffset) || 0;
    const ranges = plan.voice_ranges.map(([lo, hi]) => [lo + off - 2, hi + off + 2]);
    const order = [...Array(nv).keys()].sort((a, b) => (ranges[b][0] + ranges[b][1]) - (ranges[a][0] + ranges[a][1]));
    const rank = []; order.forEach((v, i) => { rank[v] = i; });
    const lastEnd = Math.max(...placed.map((q) => q.end));
    const total = (Math.ceil((lastEnd + 2) / 4) + 1) * 4;
    const T = Math.round(total / step);
    const fixed = Array.from({ length: nv }, () => new Array(T).fill(null));
    for (const q of placed) for (const x of q.notes) {
      const a = Math.round(x.on / step), b = Math.ceil((x.on + x.d) / step - 1e-9);
      for (let t = a; t < Math.min(b, T); t++) if (fixed[q.voice][t] === null) fixed[q.voice][t] = x.m;
    }
    const entered = [...Array(nv).keys()].map((v) => {
      const os = placed.filter((q) => q.voice === v).map((q) => Math.round(q.onset / step));
      return os.length ? Math.min(...os) : T;
    });
    const nxt = Array.from({ length: nv }, () => new Array(T).fill(null));
    for (let v = 0; v < nv; v++) {
      let cur = null;
      for (let t = T - 1; t >= 0; t--) {
        if (fixed[v][t] !== null && (t === 0 || fixed[v][t - 1] === null)) cur = [t, fixed[v][t]];
        nxt[v][t] = cur;
      }
    }
    const fixedAt = [];
    for (let t = 0; t < T; t++) { const a = []; for (let v = 0; v < nv; v++) if (fixed[v][t] !== null) a.push(fixed[v][t]); fixedAt.push(a); }
    const per = 2 * spb, nSlots = Math.floor(T / per);
    const chords = chooseChords(fixedAt, nSlots, per, key);
    const scalePcs = new Set(key.scale.map((x) => mod(key.tonic + x, 12)));
    const tonicPc = mod(key.tonic, 12);
    const melodic = (p, q, qq, hold) => {
      if (q === null) return 0;
      const d = Math.abs(p - q);
      let c = d === 0 ? 0.4 * (hold + 1) : d <= 2 ? 0 : d <= 4 ? 0.6 : d === 5 ? 1.2 : d === 6 ? 4 : d === 7 ? 1.6 : (d === 8 || d === 9 || d === 12) ? 2.5 : 5;
      if (qq !== null && Math.abs(q - qq) >= 5 && !(Math.abs(p - q) > 0 && Math.abs(p - q) <= 2 && (p - q) * (q - qq) < 0)) c += 1.5;
      if (d === 3) { const a = mod(p - key.tonic, 12), b = mod(q - key.tonic, 12); if ((a === 8 && b === 11) || (a === 11 && b === 8)) c += 3; }
      return c;
    };
    const finalBar = T - 4 * spb;
    let states = [{ c: 0, cur: new Array(nv).fill(null), prev: new Array(nv).fill(null), holds: new Array(nv).fill(0), node: null }];
    for (let t = 0; t < T; t++) {
      const ch = chords[Math.min(Math.floor(t / per), nSlots - 1)];
      const pcs = new Set([...ch.pcs].map((x) => mod(tonicPc + x, 12)));
      const root = mod(tonicPc + ch.root, 12);
      const strong = t % spb === 0;
      const free = [];
      for (let v = 0; v < nv; v++) if (t >= entered[v] && fixed[v][t] === null) free.push(v);
      const k = free.length <= 1 ? 8 : free.length === 2 ? 6 : free.length === 3 ? 5 : 4;
      const next = [];
      for (const st of states) {
        const options = [];
        for (let v = 0; v < nv; v++) {
          if (t < entered[v]) { options.push([[null, 0]]); continue; }
          if (fixed[v][t] !== null) { options.push([[fixed[v][t], 0]]); continue; }
          const q = st.cur[v], qq = st.prev[v];
          const [lo, hi] = ranges[v];
          const base = q !== null ? q : Math.round((lo + hi) / 2);
          const cands = [];
          for (let p = Math.max(lo, base - 9); p <= Math.min(hi, base + 9); p++) {
            const pc = mod(p, 12);
            if (!scalePcs.has(pc) && !pcs.has(pc)) continue;
            const hold = p === q ? st.holds[v] : 0;
            let c = melodic(p, q, qq, hold);
            if (t >= finalBar) c = q !== null ? (p === q ? 0 : 2) : 0;
            if (!pcs.has(pc)) c += strong ? 4 : (q !== null && Math.abs(p - q) <= 2 ? 0.5 : 3);
            if (rank[v] === nv - 1 && strong && t % per === 0 && pc !== root) c += 0.8;
            if (rank[v] === 0 && t === T - 1 && pc !== tonicPc) c += 3;
            const nf = nxt[v][t];
            if (nf && nf[0] - t <= 2 * spb) {
              const gap = nf[0] - t, dist = Math.abs(p - nf[1]);
              c += 0.4 * Math.max(0, dist - 2 * gap) + (gap === 1 && dist > 2 ? 0.5 : 0);
            }
            c += 0.3 * gauss(r);
            cands.push([p, c]);
          }
          cands.sort((a, b) => a[1] - b[1]);
          options.push(cands.length ? cands.slice(0, k) : [[base, 10]]);
        }
        // cartesian product over voices
        const idx = new Array(nv).fill(0);
        for (;;) {
          const pitches = new Array(nv);
          let c = st.c;
          for (let v = 0; v < nv; v++) { const o = options[v][idx[v]]; pitches[v] = o[0]; c += o[1]; }
          let bass = null;
          for (let v = 0; v < nv; v++) if (pitches[v] !== null && (bass === null || pitches[v] < bass)) bass = pitches[v];
          for (let i = 0; i < nv; i++) {
            const a = pitches[i];
            if (a === null) continue;
            for (let j = i + 1; j < nv; j++) {
              const b = pitches[j];
              if (b === null) continue;
              const fi = fixed[i][t] === null && t >= entered[i], fj = fixed[j][t] === null && t >= entered[j];
              if (!fi && !fj) continue;
              const ic = Math.abs(a - b) % 12;
              if (!CONSONANT.has(ic) && !(ic === 5 && Math.min(a, b) !== bass)) c += strong ? 3 : 0.7;
              if (a === b) c += 2;
              const hiV = rank[i] < rank[j] ? i : j, loV = hiV === i ? j : i;
              if (pitches[hiV] < pitches[loV]) c += 3;
              const ci = st.cur[i], cj = st.cur[j];
              if (ci !== null && cj !== null) {
                const da = a - ci, db = b - cj;
                if (da * db > 0 && Math.abs(ci - cj) % 12 === ic && (ic === 0 || ic === 7)) c += 6;
              }
            }
          }
          for (let q = 0; q < nv - 1; q++) {
            const u = order[q], w = order[q + 1];
            if (pitches[u] !== null && pitches[w] !== null && pitches[u] - pitches[w] > 12) c += 1;
          }
          const holds = pitches.map((x, v) => (x === st.cur[v] ? st.holds[v] + 1 : 0));
          next.push({ c, cur: pitches, prev: st.cur, holds, node: { parent: st.node, pitches } });
          let v = nv - 1;
          while (v >= 0) { idx[v]++; if (idx[v] < options[v].length) break; idx[v] = 0; v--; }
          if (v < 0) break;
        }
      }
      next.sort((a, b) => a.c - b.c);
      states = next.slice(0, beam);
      if (onProgress && t % 8 === 0) onProgress(t / T);
    }
    const grid = [];
    for (let n = states[0].node; n; n = n.parent) grid.push(n.pitches);
    grid.reverse();
    const notes = [];
    placed.forEach((q, ei) => q.notes.forEach((x) => notes.push({ v: q.voice, on: x.on, d: x.d, m: x.m, entry: ei })));
    for (let v = 0; v < nv; v++) {
      let t = 0;
      while (t < T) {
        if (fixed[v][t] !== null || grid[t][v] === null) { t++; continue; }
        const m = grid[t][v];
        let u = t + 1;
        while (u < T && fixed[v][u] === null && grid[u][v] === m && u - t < 4 * spb) u++;
        notes.push({ v, on: t * step, d: (u - t) * step, m, entry: -1 });
        t = u;
      }
    }
    notes.sort((a, b) => a.on - b.on || a.v - b.v);
    return { notes, total, chords: chords.map((c) => c.name), cost: states[0].c };
  }

  // ------------------------------------------------------------------ the whole pipeline
  function generate(text, plans, seed, onProgress) {
    const p = interpret(text, seed);
    const key = homeKey(p);
    const r = mulberry(hash(text.toLowerCase()) ^ Math.imul(seed + 1, 2654435761));
    const sub = (p.subject && parseSubject(p.subject, key)) || composeSubject(p, key, r);
    const plan = choosePlan(plans, p, r);
    const rangeOffset = sub.notes[0].m - plan.subject_first_midi;
    const placed = realise(sub, plan, key, p);
    const res = fill(placed, plan, key, { seed: seed + 7, rangeOffset, onProgress });
    return {
      request: text, seed, params: p, key: p.keyName, tempo: p.tempo, timbre: p.timbre, voices: plan.voices,
      plan: { number: plan.number, key: plan.key + " " + plan.mode, entries: plan.entries.length },
      subject: sub.notes.map((x) => ({ t: x.t, d: x.d, m: x.m })),
      entries: placed.map((q) => ({ voice: q.voice, onset: q.onset, end: q.end, inv: q.inv, localKey: q.localKey })),
      notes: res.notes, length: res.total, chords: res.chords,
    };
  }

  // ------------------------------------------------------------------ MIDI file
  function toMidi(piece) {
    const tpq = 480, bytes = [];
    const str = (s) => [...s].map((c) => c.charCodeAt(0));
    const u32 = (n) => [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255];
    const u16 = (n) => [(n >>> 8) & 255, n & 255];
    const vlq = (n) => { const out = [n & 127]; while ((n >>= 7)) out.unshift((n & 127) | 128); return out; };
    const program = { harpsichord: 6, organ: 19, piano: 0, strings: 48 }[piece.timbre] || 6;
    const tracks = [];
    const usPerQ = Math.round(60000000 / piece.tempo);
    tracks.push([0, 0xff, 0x51, 3, (usPerQ >> 16) & 255, (usPerQ >> 8) & 255, usPerQ & 255, 0, 0xff, 0x58, 4, 4, 2, 24, 8, 0, 0xff, 0x2f, 0]);
    for (let v = 0; v < piece.voices; v++) {
      const ev = [];
      for (const n of piece.notes) if (n.v === v) {
        ev.push([Math.round(n.on * tpq), 0x90 | v, n.m, n.entry >= 0 ? 84 : 72]);
        ev.push([Math.round((n.on + n.d) * tpq) - 1, 0x80 | v, n.m, 0]);
      }
      ev.sort((a, b) => a[0] - b[0] || (a[1] & 0xf0) - (b[1] & 0xf0));
      const tr = [0, 0xc0 | v, program];
      let last = 0;
      for (const e of ev) { tr.push(...vlq(e[0] - last), e[1], e[2], e[3]); last = e[0]; }
      tr.push(0, 0xff, 0x2f, 0);
      tracks.push(tr);
    }
    bytes.push(...str("MThd"), ...u32(6), ...u16(1), ...u16(tracks.length), ...u16(tpq));
    for (const tr of tracks) bytes.push(...str("MTrk"), ...u32(tr.length), ...tr);
    return new Uint8Array(bytes);
  }

  const API = { interpret, generate, toMidi, composeSubject, homeKey, choosePlan, realise, fill, keyName, mulberry, hash };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  else root.FugueEngine = API;
})(typeof self !== "undefined" ? self : this);
