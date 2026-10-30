/* Composes off the main thread so the page stays responsive. On first use it loads the learned
   models (the counterpoint network and the plan tables); if they cannot be loaded it falls back to
   the rule-based generator. */
importScripts("engine.js", "learned.js");
let ready = null;
function loadModels() {
  if (!ready) ready = (async () => {
    try {
      const base = new URL(".", self.location.href);
      const get = async (p) => { const r = await fetch(new URL(p, base)); if (!r.ok) throw new Error(p); return r; };
      const meta = await (await get("model/counterpoint.json")).json();
      const buf = await (await get("model/counterpoint.bin")).arrayBuffer();
      let pm = null;
      for (const p of ["model/plan_model.json", "results/plan_model.json", "../results/plan_model.json"]) { try { pm = await (await get(p)).json(); break; } catch (e) {} }
      if (!pm) throw new Error("plan model");
      FugueLearned.loadNet(meta, buf); FugueEngine.useLearned(FugueLearned, pm);
      return true;
    } catch (e) { return false; }
  })();
  return ready;
}
self.onmessage = async (ev) => {
  const { text, plans, seed, id, beam, rules } = ev.data;
  try {
    const learned = rules ? false : await loadModels();
    const piece = FugueEngine.generate(text, plans, seed, (f) => self.postMessage({ id, progress: f }), { beam, rules: rules || !learned });
    self.postMessage({ id, piece });
  } catch (e) {
    self.postMessage({ id, error: String(e && e.message || e) });
  }
};
