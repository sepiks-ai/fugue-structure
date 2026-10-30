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
    "string quartet": "strings", violin: "strings", violins: "strings", clavichord: "harpsichord_lute",
    "lute stop": "harpsichord_lute", "buff stop": "harpsichord_lute", "lute": "harpsichord_lute",
    "soft organ": "organ_soft", "flute stop": "organ_soft", "chamber organ": "organ_soft", "positive organ": "organ_soft" };
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
    if (["gould", "detached", "staccato", "crisp", "non-legato", "nonlegato", "pointed", "dry"].some(has)) p.articulation = "detached";
    else if (["legato", "romantic", "smooth", "sustained", "singing", "flowing"].some(has)) p.articulation = "legato";

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
    if (p.timbre !== "harpsichord") heard.push(p.timbre.replace("_lute", ", lute stop").replace("organ_soft", "soft organ"));
    if (p.articulation) heard.push(p.articulation + " touch");
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

  function fourths(a, b) {
    const at = (ns, t) => { for (const x of ns) if (x.on <= t && t < x.on + x.d) return x.m; return null; };
    let n = 0;
    for (const x of a) if (Math.abs(x.on - Math.round(x.on)) < 1e-9) { const y = at(b, x.on); if (y !== null && Math.abs(x.m - y) % 12 === 5) n++; }
    return n;
  }

  function parallels(a, b) {               // parallel fifths/octaves between two placed note lists
    const at = (ns, t) => { for (const x of ns) if (x.on <= t && t < x.on + x.d) return x.m; return null; };
    const lo = Math.max(a[0].on, b[0].on), hi = Math.min(a[a.length - 1].on + a[a.length - 1].d, b[b.length - 1].on + b[b.length - 1].d);
    let n = 0;
    for (const g of [0.25, 0.5]) {         // the quaver grid too: octaves with a semiquaver between are still heard
      let pa = null, pb = null;
      for (let t = Math.ceil(lo / g - 1e-9) * g; t < hi; t += g) {
        const x = at(a, t), y = at(b, t);
        if (x !== null && y !== null && pa !== null && pb !== null && x !== pa && (x - pa) * (y - pb) > 0) {
          const i0 = Math.abs(pa - pb) % 12, i1 = Math.abs(x - y) % 12;
          if (i0 === i1 && (i1 === 0 || i1 === 7)) n++;
        }
        pa = x; pb = y;
      }
    }
    return n;
  }

  // The exposition is built here rather than read from the plan: the analyser can miss Bach's
  // (tonal) answers, which leaves gaps of dozens of beats, or a voice that never enters.
  // Each voice enters in Bach's order, one subject-length after the last, subject and answer
  // alternating, at the octave that sits best in that voice's range. Bach's later entries follow.
  function buildEntries(sub, plan, key, p, rangeOffset) {
    const nv = plan.voices, span = sub.span, scale = span / plan.subject_span;
    const firstOn = [];
    for (let v = 0; v < nv; v++) {
      const os = plan.entries.filter((e) => e.voice === v).map((e) => e.onset);
      firstOn.push(os.length ? Math.min(...os) : Infinity);
    }
    const order = [...Array(nv).keys()].sort((a, b) => firstOn[a] - firstOn[b]);
    const ranges = plan.voice_ranges.map(([lo, hi]) => [lo + rangeOffset, hi + rangeOffset]);
    const expo = order.map((v, k) => {
      const base = k % 2 === 0 ? 0 : 7;                       // subject, answer, subject, answer
      let best = null;
      for (let o = -3; o <= 3; o++) {
        const shift = base + 12 * o;
        const ms = sub.notes.map((x) => x.m + shift);
        const [lo, hi] = ranges[v];
        const out = ms.reduce((a, m) => a + Math.max(0, lo - m, m - hi), 0);
        const centre = Math.abs((Math.min(...ms) + Math.max(...ms)) / 2 - (lo + hi) / 2);
        const sc = out * 10 + centre;
        if (!best || sc < best.sc) best = { sc, shift };
      }
      const oct = Math.floor(best.shift / 12), dia = (base === 7 ? 4 : 0) + 7 * oct;
      return { voice: v, onset: k * span, transform: "x1", midi_shift: best.shift, dia_shift: dia, expo: true };
    });
    const known = order.filter((v) => isFinite(firstOn[v]));
    const bachExpoEnd = Math.max(...known.map((v) => firstOn[v]));
    const ourExpoEnd = (order.length - 1) * span;
    const rest = plan.entries
      .filter((e) => e.onset > bachExpoEnd + 1e-9)
      .map((e) => Object.assign({}, e, { onset: ourExpoEnd + (e.onset - bachExpoEnd) * scale }));
    const maxN = p.length === "short" ? Math.max(nv + 3, 7) : p.length === "long" ? 99 : 16;
    return expo.concat(rest).slice(0, Math.max(maxN, nv));
  }

  function realise(sub, plan, key, p, rangeOffset) {
    const ents = buildEntries(sub, plan, key, p, rangeOffset || 0);
    const placed = [];
    for (const e of ents) {
      const first = !!e.expo;
      const base = Math.round(e.onset);
      const pe = placeEntry(sub, e, key);
      let best = null;
      for (const off of first ? [0, 1, 2, 3, 4] : [0, 1, -1, 2, -2, 3, -3, 4, 6, 8]) {
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
      placed.push({ kind: "subject", voice: e.voice, onset: best.on, end: best.end, notes: best.notes, transform: e.transform,
        inv: pe.inv, sc: pe.sc, localKey: pe.localKey, clash: best.clash, midi_shift: e.midi_shift, dia_shift: e.dia_shift, expo: first });
    }
    return placed;
  }

  // ------------------------------------------------------------------ 3b. countersubject
  // Written once against the subject, then brought back against later entries, like the
  // subject itself. It is kept to thirds, sixths and octaves on the beat so that it works
  // above or below the subject (invertible counterpoint at the octave); its rhythm moves
  // where the subject holds and holds where the subject runs; dissonances are passing notes,
  // neighbours or prepared suspensions that resolve down by step.
  function csRhythm(sub, p, r) {
    const span = sub.span, out = [];
    for (let b = 0; b < span - 1e-9; b += 1) {
      const len = Math.min(1, span - b);
      const c = sub.notes.filter((x) => x.t >= b - 1e-9 && x.t < b + 1 - 1e-9).length;
      let cell;
      if (len < 1) cell = [len];
      else if (c >= 3) cell = weighted(r, [[[1], 3], [[0.5, 0.5], 1]]);
      else if (c === 2) cell = weighted(r, [[[0.5, 0.5], 2], [[1], 1.5], [[0.25, 0.25, 0.5], 1]]);
      else cell = weighted(r, [[[0.25, 0.25, 0.25, 0.25], p.motion < 0 ? 0.5 : 2], [[0.5, 0.25, 0.25], 2], [[0.5, 0.5], 2], [[0.25, 0.25, 0.5], 1]]);
      out.push(...cell);
    }
    let t = 0;
    const notes = out.map((d) => { const x = { t, d }; t += d; return x; });
    if (r() < 0.5 && notes[0].d >= 0.5) { notes[0].t = 0.5; notes[0].d -= 0.5; if (notes[0].d <= 0) notes.shift(); }
    return notes;
  }

  function csPitches(sub, rh, key, r) {
    const sc = key.scale, minor = key.mode === "minor";
    const sd = sub.notes.map((x) => x.deg), sm = sub.notes.map((x) => x.m);
    const lo = Math.min(...sd) - 1, hi = Math.max(...sd) + 10;
    const centre = (Math.min(...sm) + Math.max(...sm)) / 2 + 8;
    const cands = [];
    for (let d = lo; d <= hi; d++) {
      cands.push({ deg: d, alt: 0, m: degMidi(d, 0, key.tonic, sc) });
      if (minor && (mod(d, 7) === 6 || mod(d, 7) === 5)) cands.push({ deg: d, alt: 1, m: degMidi(d, 1, key.tonic, sc) });
    }
    const K = cands.length, n = rh.length;
    const sAt = (t) => { for (const x of sub.notes) if (x.t <= t + 1e-9 && t < x.t + x.d - 1e-9) return x.m; return sm[sm.length - 1]; };
    // vertical cost of each candidate at each note, and whether its attack is dissonant
    const vert = [], dissAt = [];
    for (let i = 0; i < n; i++) {
      const { t, d } = rh[i], row = [], drow = [];
      const over = sub.notes.filter((x) => x.t < t + d - 1e-9 && x.t + x.d > t + 1e-9);
      for (const c of cands) {
        let cost = 0, attack = 0;
        for (const x of over) {
          const u = Math.max(t, x.t), strong = Math.abs(u - Math.round(u)) < 1e-9;
          const iv = c.m - x.m, ic = mod(iv, 12);
          if (iv < 0) cost += 2.5; else if (iv > 17) cost += 1.5;
          if (ic === 3 || ic === 4 || ic === 8 || ic === 9) continue;
          if (ic === 0) cost += iv === 0 ? 2.5 : 0.8;
          else if (ic === 7) cost += strong ? 1.8 : 0.5;
          else if (ic === 5) cost += strong ? 2.5 : 0.6;
          else if (u <= t + 1e-9) attack = strong ? 2 : 1;               // judged with its neighbours below
          else cost += strong ? 3 : 1.2;                                   // the subject moves into a clash
        }
        cost += 0.06 * Math.max(0, Math.abs(c.m - centre) - 5) + 0.25 * gauss(r);
        row.push(cost); drow.push(attack);
      }
      vert.push(row); dissAt.push(drow);
    }
    const mel = (a, b) => {
      const dd = Math.abs(b - a);
      return [0.8, 0, 0, 0.5, 0.5, 1, 6, 1.3, 2.2, 2.2, 5, 5, 1.8][dd] ?? 8;
    };
    const altBad = (a, b) => {                                             // a -> b, candidates
      let c = 0;
      if (a.alt && mod(a.deg, 7) === 6 && b.deg !== a.deg + 1) c += 3;     // raised leading note must rise
      if (a.alt && mod(a.deg, 7) === 5 && !(b.deg === a.deg + 1 && b.alt)) c += 3;
      if (minor && !a.alt && mod(a.deg, 7) === 6 && b.deg === a.deg + 1) c += 1.5;
      if (Math.abs(b.m - a.m) === 3 && Math.abs(b.deg - a.deg) === 1) c += 5; // augmented second
      if (a.deg === b.deg && a.alt !== b.alt) c += 4;                      // chromatic inflection
      if (b.alt) c += 0.6;                                                 // raised notes only where needed
      return c;
    };
    // the attack of note j is judged once its successor k is known (i is its predecessor)
    const judge = (i, j, k, idx) => {
      const a = dissAt[idx][j];
      if (!a) return 0;
      const inStep = i !== null && Math.abs(cands[j].m - cands[i].m) <= 2 && cands[j].m !== cands[i].m;
      const outStep = k !== null && Math.abs(cands[k].m - cands[j].m) <= 2 && cands[k].m !== cands[j].m;
      if (a === 2 && i !== null && cands[i].m === cands[j].m && k !== null && cands[k].m < cands[j].m && cands[j].m - cands[k].m <= 2) return -0.4; // suspension
      if (inStep && outStep) return a === 2 ? 2.5 : 0.2;
      return 8;
    };
    // Viterbi over (previous, current)
    let V = new Map();
    for (let j = 0; j < K; j++) V.set(-1 * K + j, { c: vert[0][j], back: null });   // prev = none (-1)
    const hist = [V];
    for (let idx = 1; idx < n; idx++) {
      const W = new Map();
      for (const [key2, st] of V) {
        const i = Math.floor(key2 / K) === -1 ? null : Math.floor(key2 / K), j = mod(key2, K);
        for (let k = 0; k < K; k++) {
          const dd = Math.abs(cands[k].m - cands[j].m);
          if (dd > 12) continue;
          let c = st.c + vert[idx][k] + mel(cands[j].m, cands[k].m) + altBad(cands[j], cands[k]) + judge(i, j, k, idx - 1);
          if (i !== null) {
            const d1 = cands[j].m - cands[i].m, d2 = cands[k].m - cands[j].m;
            if (Math.abs(d1) >= 5 && !(d2 && Math.abs(d2) <= 2 && Math.sign(d2) !== Math.sign(d1))) c += 1.5;
            if (Math.abs(d1) >= 3 && Math.abs(d2) >= 3 && Math.sign(d1) === Math.sign(d2)) c += 1.2;
            if (cands[k].m === cands[i].m && d1) c += 1.2;                 // x y x: trilling back and forth
          }
          const s0 = sAt(rh[idx].t - 0.01), s1 = sAt(rh[idx].t), cm = cands[k].m - cands[j].m, smv = s1 - s0;
          if (cm && smv && Math.sign(cm) === Math.sign(smv)) {
            c += 0.3;
            const i0 = mod(cands[j].m - s0, 12), i1 = mod(cands[k].m - s1, 12);
            if (i0 === i1 && (i1 === 0 || i1 === 7)) c += 8;
          } else if (cm && smv) c -= 0.2;
          const sp = sAt(rh[idx - 1].t), spm = s1 - sp;                    // from the previous attack, across any semiquaver
          if (cm && spm && Math.sign(cm) === Math.sign(spm)) {
            const i0 = mod(cands[j].m - sp, 12), i1 = mod(cands[k].m - s1, 12);
            if (i0 === i1 && (i1 === 0 || i1 === 7)) c += 6;
          }
          const nk = j * K + k, cur = W.get(nk);
          if (!cur || c < cur.c) W.set(nk, { c, back: key2 });
        }
      }
      // keep the search small: the best 600 (previous, current) pairs
      V = new Map([...W.entries()].sort((a, b) => a[1].c - b[1].c).slice(0, 600));
      hist.push(V);
    }
    let bestK = null, bestC = Infinity;
    for (const [k2, st] of V) {
      const j = mod(k2, K), i = Math.floor(k2 / K);
      const c = st.c + judge(i < 0 ? null : i, j, null, n - 1) + (mod(cands[j].m - sm[sm.length - 1], 12) === 0 ? 0.5 : 0);
      if (c < bestC) { bestC = c; bestK = k2; }
    }
    if (bestK === null) return null;
    const seq = [];
    for (let idx = n - 1, k2 = bestK; idx >= 0; idx--) { seq.push(mod(k2, K)); k2 = hist[idx].get(k2).back; }
    seq.reverse();
    // tie repeated pitches into one longer note (this is how the suspensions are written)
    const notes = [];
    seq.forEach((j, idx) => {
      const c = cands[j], last = notes[notes.length - 1];
      if (last && last.m === c.m && Math.abs(last.t + last.d - rh[idx].t) < 1e-9) last.d += rh[idx].d;
      else notes.push({ t: rh[idx].t, d: rh[idx].d, deg: c.deg, alt: c.alt, m: c.m });
    });
    return { notes, span: sub.span, c: bestC / n };
  }

  function composeCountersubject(sub, p, key, r) {
    let best = null;
    for (let k = 0; k < 12; k++) {
      const res = csPitches(sub, csRhythm(sub, p, r), key, r);
      if (!res || res.notes.length < 4) continue;
      const on = (ns) => ns.map((x) => ({ on: x.t, d: x.d, m: x.m }));
      res.c += 3 * parallels(on(res.notes), on(sub.notes)) + 0.5 * fourths(on(sub.notes), on(res.notes));
      if (!best || res.c < best.c) best = res;
    }
    return best;
  }

  // shared scoring for lines dropped into the texture: clashes, parallels, range, crossings
  function lineScore(notes, voice, others, ranges, rank) {
    const [lo, hi] = ranges[voice];
    let s = 3 * notes.reduce((a, x) => a + Math.max(0, lo - x.m, x.m - hi), 0);
    const on = notes[0].on, end = notes[notes.length - 1].on + notes[notes.length - 1].d;
    for (const q of others) {
      if (!(on < q.end && q.onset < end)) continue;
      s += strongClashes(notes, q.notes) + 0.5 * fourths(notes, q.notes) + 3 * parallels(notes, q.notes);
      const above = rank[voice] < rank[q.voice];
      for (const x of notes) for (const y of q.notes) if (y.on <= x.on && x.on < y.on + y.d && (above ? x.m < y.m : x.m > y.m)) s += 1.5;
    }
    return s;
  }

  function voiceRanks(plan, rangeOffset) {
    const ranges = plan.voice_ranges.map(([lo, hi]) => [lo + rangeOffset, hi + rangeOffset]);
    const order = [...Array(plan.voices).keys()].sort((a, b) => (ranges[b][0] + ranges[b][1]) - (ranges[a][0] + ranges[a][1]));
    const rank = []; order.forEach((v, i) => { rank[v] = i; });
    return { ranges, rank };
  }

  function addCountersubjects(cs, placed, key, plan, rangeOffset, r) {
    if (!cs) return [];
    const { ranges, rank } = voiceRanks(plan, rangeOffset);
    const items = [];
    const busy = (v, a, b) => placed.concat(items).some((q) => q.voice === v && a < q.end - 1e-9 && q.onset < b - 1e-9);
    placed.forEach((e, i) => {
      if (i === 0 || e.inv || e.sc !== 1) return;
      if (!e.expo && r() > 0.85) return;                                  // Bach does not bring it back every time
      let best = null;
      for (let v = 0; v < plan.voices; v++) {
        if (v === e.voice || busy(v, e.onset, e.end)) continue;
        // the voice that has just finished the subject carries on into the countersubject
        const justDone = placed.some((q) => q.voice === v && Math.abs(q.end - e.onset) <= 1);
        if (e.expo && !justDone) continue;
        for (let o = -3; o <= 3; o++) {
          const pe = placeEntry(cs, { transform: "x1", midi_shift: e.midi_shift + 12 * o, dia_shift: e.dia_shift + 7 * o }, key);
          const notes = pe.notes.map((x) => ({ on: e.onset + x.t, d: x.d, m: x.m }));
          const s = lineScore(notes, v, placed.concat(items), ranges, rank) + (justDone ? -1 : 0) + 0.3 * r();
          if (!best || s < best.s) best = { s, v, notes };
        }
      }
      if (!best || best.s > 4) return;
      const end = best.notes[best.notes.length - 1].on + best.notes[best.notes.length - 1].d;
      items.push({ kind: "cs", voice: best.v, onset: best.notes[0].on, end, notes: best.notes, against: i });
    });
    return items;
  }

  // ------------------------------------------------------------------ 3c. episodes
  // Between groups of entries Bach rarely writes free music: he takes a fragment of the subject
  // or countersubject and moves it through a sequence, usually stepping down, with a second voice
  // imitating it a little later. The same here, in the gaps the plan leaves.
  function motifFrom(line, fromEnd) {
    const ns = line.notes;
    let pick = [];
    if (!fromEnd) {
      for (const x of ns) { pick.push(x); const len = x.t + x.d - ns[0].t; if (pick.length >= 4 && len >= 1) break; }
    } else {
      for (let i = ns.length - 2; i >= 0; i--) { pick.unshift(ns[i]); const len = ns[ns.length - 2].t + ns[ns.length - 2].d - pick[0].t; if (pick.length >= 4 && len >= 1) break; }
    }
    const t0 = pick[0].t, len = pick[pick.length - 1].t + pick[pick.length - 1].d - t0;
    const P = len <= 1 ? 1 : len <= 2 ? 2 : 4;
    const notes = pick.filter((x) => x.t - t0 < P).map((x) => ({ t: x.t - t0, d: x.d, deg: x.deg }));
    const last = notes[notes.length - 1]; last.d = P - last.t;
    return { notes, P };
  }

  function seqNotes(motif, start, reps, step, base, key, until) {
    const out = [];
    for (let k = 0; k < reps; k++) for (const x of motif.notes) {
      const on = start + k * motif.P + x.t;
      if (on + x.d > until + 1e-9) continue;
      out.push({ on, d: x.d, deg: x.deg + base + k * step });
    }
    for (let i = 0; i < out.length; i++) {
      let alt = 0;
      const nx = out[i + 1];
      if (key.mode === "minor" && mod(out[i].deg, 7) === 6 && nx && nx.deg === out[i].deg + 1) alt = 1;
      out[i].m = degMidi(out[i].deg, alt, key.tonic, key.scale);
    }
    return out.map(({ on, d, m }) => ({ on, d, m }));
  }

  function addEpisodes(sub, cs, placed, extra, key, plan, rangeOffset, r) {
    const { ranges, rank } = voiceRanks(plan, rangeOffset);
    const ents = placed.slice().sort((a, b) => a.onset - b.onset);
    const expoEnd = Math.max(...placed.filter((q) => q.expo).map((q) => q.end));
    const gaps = [];
    let cur = -Infinity;
    for (const e of ents) {
      if (cur >= expoEnd - 1e-9 && e.onset - cur >= 3) gaps.push([cur, e.onset]);
      cur = Math.max(cur, e.end);
    }
    const motifs = [motifFrom(sub, false), motifFrom(sub, true)];
    if (cs) motifs.push(motifFrom(cs, false));
    const all = placed.concat(extra), items = [];
    const lastPitch = (v, t) => {
      let best = null;
      for (const q of all) if (q.voice === v) for (const x of q.notes) if (x.on < t && (!best || x.on > best.on)) best = x;
      return best ? best.m : null;
    };
    for (const [a, b] of gaps) {
      const start = Math.ceil(a - 1e-9);
      let best = null;
      for (let trial = 0; trial < 40; trial++) {
        const motif = pick(r, motifs), P = motif.P;
        const reps = Math.min(4, Math.floor((b - start) / P));
        if (reps < 2) continue;
        const step = weighted(r, [[-1, 4], [1, 2], [-2, 1]]);
        const A = Math.floor(r() * plan.voices);
        let B = Math.floor(r() * plan.voices); if (B === A) B = (A + 1) % plan.voices;
        const delay = P === 4 ? pick(r, [1, 2]) : P / 2;           // never a whole period: that is parallel motion
        const fit = (v, base, st0) => {
          const ns = seqNotes(motif, st0, reps, step, base, key, b);
          return ns.length ? ns : null;
        };
        // put the leader where its voice can carry the whole sequence, near where it last was
        let bestA = null;
        for (let base = -16; base <= 16; base++) {
          const ns = fit(A, base, start);
          if (!ns) continue;
          const [lo, hi] = ranges[A], lp = lastPitch(A, start);
          const s = ns.reduce((x, y) => x + Math.max(0, lo - y.m, y.m - hi), 0) * 3 + (lp === null ? 0 : 0.15 * Math.abs(ns[0].m - lp));
          if (!bestA || s < bestA.s) bestA = { s, base, ns };
        }
        if (!bestA) continue;
        const lineA = { voice: A, onset: start, end: b, notes: bestA.ns };
        let s = lineScore(bestA.ns, A, all, ranges, rank);
        let lineB = null, sB = 2 * reps;
        if (plan.voices > 1) for (const dI of [-7, -2, -5, 2, 5, 7, -9, -14, -4, 3]) {
          const nsB = fit(B, bestA.base + dI, start + delay);
          if (!nsB || nsB.length < motif.notes.length) continue;
          const sb = lineScore(nsB, B, all.concat([lineA]), ranges, rank);
          if (sb < sB) { sB = sb; lineB = { voice: B, onset: nsB[0].on, end: b, notes: nsB }; }
        }
        s = (s + sB) / reps + 0.3 * r();
        if (!best || s < best.s) best = { s, lines: lineB ? [lineA, lineB] : [lineA] };
      }
      if (!best || best.s > 1) continue;
      for (const L of best.lines) {
        const end = L.notes[L.notes.length - 1].on + L.notes[L.notes.length - 1].d;
        const it = { kind: "episode", voice: L.voice, onset: L.notes[0].on, end, notes: L.notes };
        items.push(it); all.push(it);
      }
    }
    return items;
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
              // a pair is judged if either voice is free now or was free a moment ago (moving into fixed material)
              const freeAt = (v, u) => u >= 0 && fixed[v][u] === null && u >= entered[v];
              const fi = freeAt(i, t) || freeAt(i, t - 1), fj = freeAt(j, t) || freeAt(j, t - 1);
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
    let ei = 0;
    placed.forEach((q) => {
      const idx = q.kind === "subject" || !q.kind ? ei++ : -1;
      q.notes.forEach((x) => notes.push({ v: q.voice, on: x.on, d: x.d, m: x.m, entry: idx, kind: q.kind || "subject" }));
    });
    for (let v = 0; v < nv; v++) {
      let t = 0;
      while (t < T) {
        if (fixed[v][t] !== null || grid[t][v] === null) { t++; continue; }
        const m = grid[t][v];
        let u = t + 1;
        while (u < T && fixed[v][u] === null && grid[u][v] === m && u - t < 4 * spb) u++;
        notes.push({ v, on: t * step, d: (u - t) * step, m, entry: -1, kind: "free" });
        t = u;
      }
    }
    notes.sort((a, b) => a.on - b.on || a.v - b.v);
    return { notes, total, chords: chords.map((c) => c.name), cost: states[0].c };
  }

  // ------------------------------------------------------------------ the whole pipeline
  function generate(text, plans, seed, onProgress, opts) {
    if (LEARNED && !(opts && opts.rules)) return generateLearned(text, seed, onProgress, opts);
    const beam = (opts && opts.beam) || 30;
    const p = interpret(text, seed);
    const key = homeKey(p);
    const r = mulberry(hash(text.toLowerCase()) ^ Math.imul(seed + 1, 2654435761));
    const sub = (p.subject && parseSubject(p.subject, key)) || composeSubject(p, key, r);
    const plan = choosePlan(plans, p, r);
    const rangeOffset = sub.notes[0].m - plan.subject_first_midi;
    const placed = realise(sub, plan, key, p, rangeOffset);
    const cs = composeCountersubject(sub, p, key, r);
    const csItems = addCountersubjects(cs, placed, key, plan, rangeOffset, r);
    const epItems = addEpisodes(sub, cs, placed, csItems, key, plan, rangeOffset, r);
    const res = fill(placed.concat(csItems, epItems), plan, key, { seed: seed + 7, rangeOffset, onProgress, beam });
    return {
      request: text, seed, params: p, key: p.keyName, tempo: p.tempo, timbre: p.timbre, voices: plan.voices,
      plan: { number: plan.number, key: plan.key + " " + plan.mode, entries: plan.entries.length },
      subject: sub.notes.map((x) => ({ t: x.t, d: x.d, m: x.m })),
      countersubject: cs ? cs.notes.map((x) => ({ t: x.t, d: x.d, m: x.m })) : null,
      counts: { countersubjects: csItems.length, episodes: epItems.length },
      entries: placed.map((q) => ({ voice: q.voice, onset: q.onset, end: q.end, inv: q.inv, localKey: q.localKey })),
      notes: res.notes, length: res.total, chords: res.chords,
    };
  }

  // ------------------------------------------------------------------ the learned generator
  // Everything between the thematic material is written by the counterpoint model; the plan comes from
  // the learned plan model; the countersubject is the line the model writes against the answer.
  let LEARNED = null;                                    // { L: learned.js API, PM: plan model }
  function useLearned(L, PM) { LEARNED = { L, PM }; }
  function learnedPlan(sp, sub, key, nv, PM) {
    const ms = sub.notes.map((x) => x.m - sub.notes[0].m), subMid = (Math.min(...ms) + Math.max(...ms)) / 2;
    const R = (PM.ranges[String(nv)] || PM.ranges["3"]).slice().sort((a, b) => a[0] - b[0]).map(([, lo, hi]) => [lo, hi]);
    const FORM = { plain: "x1", inv: "INV+x1", aug: "AUG", dim: "DIM" };
    const entries = sp.entries.map((e) => {
      const [lo, hi] = R[e.rank], semis = key.scale[e.degree];
      let best = 0, bd = 1e9;
      for (let k = -3; k <= 3; k++) { const dd = Math.abs(semis + 12 * k + subMid - (lo + hi) / 2); if (dd < bd) { bd = dd; best = k; } }
      return { voice: e.rank, onset: e.onset, transform: FORM[e.form] || "x1", dia_shift: e.degree + 7 * best, midi_shift: semis + 12 * best };
    });
    return { number: null, learned: true, key: "", mode: key.mode, voices: nv, voice_ranges: R, subject_span: 1, subject_first_midi: 0, entries };
  }
  function lineToSubject(notes, t0, span, key, shift) {   // a heard line -> a line in the home key, as the subject is stored
    const out = [];
    for (const x of notes) {
      const midi = x.m - shift, rel = midi - key.tonic, oct = Math.floor(rel / 12), pc = mod(rel, 12);
      let deg = 0; for (let i = 6; i >= 0; i--) if (key.scale[i] <= pc) { deg = i; break; }
      out.push({ t: x.on - t0, d: x.d, deg: deg + 7 * oct, alt: pc - key.scale[deg], m: midi });
    }
    return out.length >= 4 ? { notes: out, span } : null;
  }
  // a subject written by the model: it plays the opening of a fugue alone, as Bach's first voice does, many times
  // over; the lines are cut where a phrase can end and the one that best fits the description is kept.
  // Lengths come from the 48 subjects (in crotchets); everything about the line itself comes from the model.
  const SUBJECT_SPANS = [[4, 6], [6, 9], [8, 11], [12, 13]];
  let SUBJ_TEMP = 0.75, SUBJ_NUC = 0.93, SUBJ_N = 48;
  function subjectSampling(t, nuc, n) { SUBJ_TEMP = t; SUBJ_NUC = nuc; SUBJ_N = n; }
  function learnedSubject(p, key, nv, voice, r, seed) {
    const { L } = LEARNED;
    const spans = p.length === "short" ? [[4, 6], [6, 9]] : p.length === "long" ? [[8, 11], [12, 13]] : SUBJECT_SPANS;
    const triad = new Set([0, key.mode === "minor" ? 3 : 4, 7].map((x) => mod(key.tonic + x, 12)));
    const SUBJ = LEARNED.PM.subject;
    let best = null;
    let tried = 0, kept = 0; const why = {};
    for (let c = 0; c < SUBJ_N; c++) {
      const span = weighted(r, spans); tried++;
      const first = r() < 0.65 ? key.tonic : key.tonic + 7 > 67 ? key.tonic - 5 : key.tonic + 7;
      const T = Math.round((span + 2) / 0.25);
      const res = L.compose({ voices: nv, placed: [{ voice, onset: 0, end: 0.25, notes: [{ on: 0, d: 0.25, m: first }] }], total: T * 0.25,
        tonic: key.tonic, mode: key.mode, ranges: [...Array(nv)].map(() => [first - 9, first + 14]),
        seed: (seed * 7907 + c * 131 + 17) >>> 0, particles: 1, sweeps: 0, temperature: SUBJ_TEMP, nucleus: SUBJ_NUC, solo: true });
      // read the line back off the grid
      const P = res.pitch[voice], O = res.onset[voice], line = [];
      let rest = false;
      for (let t = 0; t < T; t++) {
        if (P[t] < 0) { if (t * 0.25 < span) rest = true; continue; }
        if (t === 0 || O[t] || P[t] !== P[t - 1]) line.push({ on: t * 0.25, d: 0.25, m: P[t] }); else line[line.length - 1].d += 0.25;
      }
      if (rest) { why.rest = (why.rest || 0) + 1; continue; }
      // where can it end? on a beat, on a note of the tonic triad, near the length chosen; the last note is held
      // to the next beat at least (the answer enters after it)
      let cut = -1;
      for (let i = line.length - 1; i >= 4; i--) {
        const e = line[i].on;
        if (e % 1 !== 0 || e > span - 0.5 || e < span - 2.5 || !triad.has(mod(line[i].m, 12))) continue;
        if (cut < 0 || Math.abs(e - (span - 1)) < Math.abs(line[cut].on - (span - 1))) cut = i;
      }
      if (cut < 0) { why.cut = (why.cut || 0) + 1; continue; }
      const notes = line.slice(0, cut + 1).map((x) => Object.assign({}, x));
      notes[cut].d = Math.max(1, Math.min(2, Math.ceil(notes[cut].d)));
      const spanHere = notes[cut].on + notes[cut].d;
      const m = notes.map((x) => x.m), ints = m.slice(1).map((x, i) => x - m[i]);
      const range = Math.max(...m) - Math.min(...m);
      if (notes.length < 5 || notes.length > 24 || range < 5 || range > 16 || ints.some((x) => Math.abs(x) > 12)) { why.shape = (why.shape || 0) + 1; continue; }
      let run = 1, bad = false; for (let i = 1; i < m.length; i++) { run = m[i] === m[i - 1] ? run + 1 : 1; if (run > 2) bad = true; }
      if (bad) { why.repeat = (why.repeat || 0) + 1; continue; }
      const sub = lineToSubject(notes, 0, spanHere, key, 0);
      if (!sub) continue;
      // how like Bach's subjects it is (measures learned from the 48, scripts/learn_subjects.py), with the words
      // moving the targets: lively asks for more notes, angular for more leaps; and how sure the model was
      const steps = ints.filter((x) => Math.abs(x) <= 2).length / ints.length;
      const meas = { log_density: Math.log(notes.length / spanHere), range, steps, leaps: ints.filter((x) => Math.abs(x) >= 5).length / ints.length,
        lengths: new Set(notes.map((x) => x.d)).size, longest: Math.max(...notes.map((x) => x.d)) / spanHere,
        peak_at: m.indexOf(Math.max(...m)) / (m.length - 1) };
      const shift = { log_density: 0.45 * p.motion, steps: -0.15 * p.leap, leaps: 0.1 * p.leap };
      const minorOk = (x) => key.mode === "minor" && x.alt === 1 && [5, 6].includes(mod(x.deg, 7));
      const alts = sub.notes.filter((x) => x.alt !== 0 && !minorOk(x)).length;
      let sc = res.logp / Math.max(1, res.n);
      for (const k in meas) { const [mu, sd] = SUBJ.measures[k]; const z = (meas[k] - mu - (shift[k] || 0)) / sd; sc -= 0.3 * z * z; }
      sc += p.chromatic ? 0.4 * Math.min(3, alts) : -0.3 * alts;
      if (ints.some((x) => Math.abs(x) === 6)) sc -= 1;
      sc += 0.15 * gauss(r);
      kept++;
      if (!best || sc > best.sc) best = { sc, sub };
    }
    if (typeof process !== "undefined" && process.env && process.env.SUBJ_DEBUG) console.log("subject candidates kept", kept, "of", tried, JSON.stringify(why));
    return best && best.sub;
  }
  function generateLearned(text, seed, onProgress, opts) {
    const { L, PM } = LEARNED;
    const p = interpret(text, seed);
    const key = homeKey(p);
    const r = mulberry(hash(text.toLowerCase()) ^ Math.imul(seed + 1, 2654435761));
    const nv = Math.max(2, Math.min(5, p.voices));
    const sp = L.samplePlan(PM, nv, r, { length: p.length, stretto: p.stretto, inversion: p.inversion });
    const given = p.subject && parseSubject(p.subject, key);
    const heard = given || (opts && opts.ruleSubject) || !PM.subject ? null : learnedSubject(p, key, nv, sp.entries[0].rank, r, seed);
    const sub = given || heard || composeSubject(p, key, r);
    const plan = learnedPlan(sp, sub, key, nv, PM);
    const rangeOffset = sub.notes[0].m;
    const placed = realise(sub, plan, key, p, rangeOffset);
    const ranges = plan.voice_ranges.map(([lo, hi]) => [lo + rangeOffset - 2, hi + rangeOffset + 2]);
    const particles = (opts && opts.beam) >= 100 ? 16 : 8, sweeps = 2;
    // sampling settings, chosen by measuring the results against the 48 fugues (rate of suspensions, of clashes
    // left untreated, and of quick notes): pitch choices a little sharpened, a nudge to move on between the beats,
    // and no leaping into a clash on the beat
    const STYLE = { temperature: 0.85, nucleus: 0.95, moveBias: 0.6, strictBeats: true };
    const prog = (a, b) => (f) => onProgress && onProgress(a + (b - a) * f);
    // 1. the exposition's opening: the model writes the first voice's line against the answer
    let cs = null;
    const e1 = placed[0], e2 = placed[1];
    if (e2) {
      const head = L.compose({ voices: nv, placed: placed.filter((q) => q.onset < e2.end - 1e-9), total: e2.end, tonic: key.tonic, mode: key.mode,
        ranges, seed: seed * 7919 + 13, particles, sweeps, ...STYLE, onProgress: prog(0, 0.25) });
      const line = head.notes.filter((n) => n.v === e1.voice && n.on >= e2.onset - 1e-9 && n.on < e2.end - 1e-9);
      cs = lineToSubject(line, e2.onset, sub.span, key, e2.midi_shift);
    }
    const csItems = addCountersubjects(cs, placed, key, plan, rangeOffset, r);
    const epItems = addEpisodes(sub, cs, placed, csItems, key, plan, rangeOffset, r);
    const fixed = placed.concat(csItems, epItems);
    const lastEnd = Math.max(...fixed.map((q) => q.end));
    const total = (Math.ceil((lastEnd + 2) / 4) + 1) * 4;
    // 2. everything else
    const res = L.compose({ voices: nv, placed: fixed, total: total - 4, tonic: key.tonic, mode: key.mode, ranges, seed: seed * 104729 + 7,
      particles, sweeps, ...STYLE, onProgress: prog(0.25, 1) });
    // 3. a final chord: each voice moves to the nearest note of the tonic triad (major, as Bach ends most minor fugues)
    const notes = [];
    let ei = 0;
    for (const q of fixed) {
      const idx = q.kind === "subject" || !q.kind ? ei++ : -1;
      for (const x of q.notes) notes.push({ v: q.voice, on: x.on, d: x.d, m: x.m, entry: idx, kind: q.kind || "subject" });
    }
    notes.push(...res.notes);
    const triad = [0, 4, 7].map((x) => mod(key.tonic + x, 12));
    const order = [...Array(nv).keys()];
    let lastBass = null;
    for (const v of order.slice().reverse()) {
      const mine = notes.filter((n) => n.v === v && n.on < total - 4);
      const last = mine.length ? mine.reduce((a, b) => (a.on > b.on ? a : b)) : null;
      if (!last) continue;
      if (last.on + last.d > total - 4) last.d = total - 4 - last.on;
      let best = null;
      for (let m = last.m - 7; m <= last.m + 7; m++) {
        const pc = mod(m, 12); if (!triad.includes(pc)) continue;
        if (v === nv - 1 && pc !== mod(key.tonic, 12)) continue;      // the bass ends on the tonic
        if (!best || Math.abs(m - last.m) < Math.abs(best - last.m)) best = m;
      }
      notes.push({ v, on: total - 4, d: 4, m: best, entry: -1, kind: "free" });
      if (v === nv - 1) lastBass = best;
    }
    void lastBass;
    notes.sort((a, b) => a.on - b.on || a.v - b.v);
    return {
      request: text, seed, params: p, key: p.keyName, tempo: p.tempo, timbre: p.timbre, voices: nv,
      plan: { number: null, learned: true, key: p.keyName, entries: placed.length },
      subject: sub.notes.map((x) => ({ t: x.t, d: x.d, m: x.m })),
      countersubject: cs ? cs.notes.map((x) => ({ t: x.t, d: x.d, m: x.m })) : null,
      counts: { countersubjects: csItems.length, episodes: epItems.length },
      entries: placed.map((q) => ({ voice: q.voice, onset: q.onset, end: q.end, inv: q.inv, localKey: q.localKey })),
      notes, length: total, chords: [], learned: true, subjectLearned: !!heard,
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

  const API = { interpret, generate, generateLearned, useLearned, subjectSampling, toMidi, composeSubject, homeKey, choosePlan, realise, fill, keyName, mulberry, hash };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  else root.FugueEngine = API;
})(typeof self !== "undefined" ? self : this);
