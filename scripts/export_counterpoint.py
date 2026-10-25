"""Export the trained counterpoint model for the browser: web/model/counterpoint.bin + counterpoint.json.
Embedding rows are stored as int8 with one float scale per row; the two small layers as float32."""
import sys, json, os
import numpy as np, torch
sys.path.insert(0, "src")
from fugue.learn.model import Counterpoint
from fugue.learn.grid import TABLES, NAMES

src = sys.argv[1] if len(sys.argv) > 1 else "results/counterpoint_all.pt"
m = Counterpoint(); m.load_state_dict(torch.load(src)); m.eval()
E = m.emb.weight.detach().numpy().astype(np.float32); R, H = E.shape
sc = np.maximum(np.abs(E).max(1), 1e-8) / 127.0
q = np.clip(np.round(E / sc[:, None]), -127, 127).astype(np.int8)
parts = [q.tobytes()]
pad = (-len(parts[0])) % 4; parts.append(b"\0" * pad)
for a in (sc.astype(np.float32), m.b0.detach().numpy(), m.l1.weight.detach().numpy(), m.l1.bias.detach().numpy(),
          m.out.weight.detach().numpy(), m.out.bias.detach().numpy()):
    parts.append(np.ascontiguousarray(a, dtype=np.float32).tobytes())
os.makedirs("web/model", exist_ok=True)
open("web/model/counterpoint.bin", "wb").write(b"".join(parts))
json.dump({"hidden": H, "rows": R, "tables": [[n, TABLES[n]] for n in NAMES], "trained_on": "48 fugues, Kyle Rother's open edition of the Well-Tempered Clavier (CC BY 4.0)", "source": os.path.basename(src)},
          open("web/model/counterpoint.json", "w"))
print("rows", R, "hidden", H, "bytes", sum(len(p) for p in parts))
