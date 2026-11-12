// The browser's token composer (web/tx.js) against the Python model: same logits on 300 tokens of WTC I/2.
const fs = require("fs"), path = require("path"), assert = require("assert");
const Tx = require("../web/tx.js");
const meta = JSON.parse(fs.readFileSync(path.join(__dirname, "../web/tx/model.json"))), b = fs.readFileSync(path.join(__dirname, "../web/tx/model.bin"));
Tx.init(meta, b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
const P = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures/tx_parity.json")));
const c = Tx.Cache(400), out = {};
P.ids.forEach((t, i) => { const lg = Tx.step(c, t, P.aux[i], i < 5); if ([10, 100, 299].includes(i)) out[i] = lg; });
let worst = 0; [10, 100, 299].forEach((i, k) => P.q[k].forEach((v, j) => (worst = Math.max(worst, Math.abs(out[i][j] - v)))));
assert.ok(worst < 1e-3, "logits differ by " + worst);
const plan = JSON.parse(fs.readFileSync(path.join(__dirname, "../web/model/plan_model.json")));
const p = Tx.samplePlan(plan, 3, () => 0.5, {});
assert.ok(p.entries.length >= 4 && p.order.length === 3, "plan has an exposition and later entries");
console.log("tx ok: logits within", worst.toExponential(1), "of Python;", p.entries.length, "planned entries");
