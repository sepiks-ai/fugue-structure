/* Composes off the main thread so the page stays responsive. */
importScripts("engine.js");
self.onmessage = (ev) => {
  const { text, plans, seed, id, beam } = ev.data;
  try {
    const piece = FugueEngine.generate(text, plans, seed, (f) => self.postMessage({ id, progress: f }), { beam });
    self.postMessage({ id, piece });
  } catch (e) {
    self.postMessage({ id, error: String(e && e.message || e) });
  }
};
