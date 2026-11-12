"""Export the token composer's weights for the browser: web/tx/model.bin and web/tx/model.json.

    python scripts/export_tx.py results/tx4/best.pt

Matrices (embeddings and linear weights) are stored as int8 with one float32 scale per row; biases and
layer norms as float32. model.json carries the configuration, the vocabulary and where each tensor is."""
import json, sys, os, struct, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fugue.tx import tokens as T

def main(path, out="web/tx"):
    ck = torch.load(path, map_location="cpu"); sd = ck["model"]; os.makedirs(out, exist_ok=True)
    blobs, index, off = [], {}, 0
    for name, t in sd.items():
        a = t.float().numpy()
        if a.ndim == 2:
            scale = np.maximum(np.abs(a).max(1), 1e-12) / 127.0
            q = np.clip(np.round(a / scale[:, None]), -127, 127).astype(np.int8)
            sb = scale.astype(np.float32).tobytes(); qb = q.tobytes()
            pad = (-(off + len(sb) + len(qb))) % 4
            index[name] = {"shape": list(a.shape), "int8": True, "scale": off, "data": off + len(sb)}
            blobs += [sb, qb, b"\0" * pad]; off += len(sb) + len(qb) + pad
        else:
            b = a.astype(np.float32).tobytes()
            index[name] = {"shape": list(a.shape), "int8": False, "data": off}; blobs.append(b); off += len(b)
    open(os.path.join(out, "model.bin"), "wb").write(b"".join(blobs))
    json.dump({"cfg": ck["cfg"], "vocab": T.VOCAB, "tpb": T.TPB, "lo": T.LO, "hi": T.HI, "max_int": T.MAX_INT,
               "durs": T.DURS, "tensors": index, "source": f"{os.path.basename(os.path.dirname(path))}, step {ck.get('step')}, held-out loss {ck.get('val'):.4f}"},
              open(os.path.join(out, "model.json"), "w"))
    print(f"{off / 1e6:.2f} MB, {len(index)} tensors")

if __name__ == "__main__":
    main(*sys.argv[1:])
