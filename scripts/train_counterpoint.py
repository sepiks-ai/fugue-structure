"""Train the counterpoint model on the open edition. usage: python scripts/train_counterpoint.py [--all]
Without --all, six fugues are held out and the held-out loss is reported every epoch."""
import sys, time, json, math
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "src")
from fugue.learn.dataset import load, build
from fugue.learn.model import Counterpoint
from fugue.learn.grid import HOLD, REST

HELD = {"wtc1f02", "wtc1f11", "wtc1f20", "wtc2f05", "wtc2f14", "wtc2f23"}
torch.manual_seed(0); torch.set_num_threads(2)
fugues = load(); use_all = "--all" in sys.argv
train = [f for f in fugues if use_all or f["piece"] not in HELD]; held = [f for f in fugues if f["piece"] in HELD]
t0 = time.time()
Xtr, Ytr = build(train, seed=1, copies=3); Xte, Yte = build(held, seed=99, copies=1)
print(f"train {len(Ytr)} examples, held-out {len(Yte)} ({time.time() - t0:.0f}s)", flush=True)
Xtr, Ytr, Xte, Yte = map(torch.from_numpy, (Xtr, Ytr.astype(np.int64), Xte, Yte.astype(np.int64)))
model = Counterpoint()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=2e-2)
EPOCHS = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--epochs=")), 8)); B = 1024
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS * math.ceil(len(Ytr) / B))
def evaluate():
    model.eval()
    with torch.no_grad():
        lo = F.cross_entropy(model(Xte), Yte).item(); pred = model(Xte).argmax(1)
        on = (Yte != HOLD) & (Yte != REST)
        acc = (pred == Yte).float().mean().item(); acc_on = (pred[on] == Yte[on]).float().mean().item()
    model.train(); return lo, acc, acc_on
best = (1e9, 0, None)
for ep in range(EPOCHS):
    perm = torch.randperm(len(Ytr)); tot = 0
    for i in range(0, len(Ytr), B):
        idx = perm[i:i + B]; loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(idx)
    lo, acc, acc_on = evaluate() if held else (float("nan"),) * 3
    if held and lo < best[0]: best = (lo, ep + 1, {k: v.clone() for k, v in model.state_dict().items()})
    print(f"epoch {ep + 1}: train {tot / len(Ytr):.3f}  held-out {lo:.3f} (perplexity {math.exp(lo):.2f}), accuracy {acc:.1%}, next-pitch accuracy {acc_on:.1%}  [{time.time() - t0:.0f}s]", flush=True)
if best[2] is not None: model.load_state_dict(best[2]); print(f"kept epoch {best[1]} (held-out {best[0]:.3f})")
torch.save(model.state_dict(), "results/counterpoint_all.pt" if use_all else "results/counterpoint.pt")
