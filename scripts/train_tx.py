"""Train the token composer.

    py scripts/train_tx.py [--ctx=2048] [--batch=8] [--steps=20000] [--d=384] [--layers=8] [--heads=6] [--lr=3e-4]
                           [--out=results/tx] [--resume] [--cpu]

Uses every GPU it finds (two 1080 Ti: batch split across both with DataParallel, which works on Windows).
Pascal cards are slow at half precision, so it trains in float32. Every --eval steps it reports held-out loss
on the six WTC fugues and the Mutopia pieces set aside, keeps the best weights in <out>/best.pt and the latest
in <out>/last.pt (resumable)."""
import os, sys, time, math, json, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fugue.tx import tokens as T
from fugue.tx.model import Composer
from fugue.tx.data import Windows

def args():
    a = dict(ctx=2048, batch=8, steps=20000, d=384, layers=8, heads=6, lr=3e-4, out="results/tx", eval=500, warm=500, drop=0.1, data="data/tx_pieces.json")
    flags = set()
    for x in sys.argv[1:]:
        if "=" in x: k, v = x[2:].split("=", 1); a[k] = type(a[k])(v) if k in a else v
        else: flags.add(x[2:])
    return a, flags

def main():
    a, flags = args(); os.makedirs(a["out"], exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() and "cpu" not in flags else "cpu"
    ngpu = torch.cuda.device_count() if dev == "cuda" else 0
    print(f"device {dev}, GPUs {ngpu}: {[torch.cuda.get_device_name(i) for i in range(ngpu)]}", flush=True)
    model = Composer(len(T.VOCAB), a["d"], a["layers"], a["heads"], a["drop"]).to(dev)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a["lr"], betas=(0.9, 0.95), weight_decay=0.1)
    step, best = 0, 1e9
    if "resume" in flags and os.path.exists(os.path.join(a["out"], "last.pt")):
        ck = torch.load(os.path.join(a["out"], "last.pt"), map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step, best = ck["step"], ck["best"]
        print("resumed at step", step, flush=True)
    run = torch.nn.DataParallel(model) if ngpu > 1 else model
    train, val = Windows(a["data"], "train", a["ctx"], seed=step), Windows(a["data"], "val", a["ctx"], seed=1)
    vb = [val.batch(a["batch"]) for _ in range(8)]
    lr_at = lambda s: a["lr"] * min(1, (s + 1) / a["warm"]) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / a["steps"]))))
    t0, tok = time.time(), 0
    while step < a["steps"]:
        model.train()
        for g in opt.param_groups: g["lr"] = lr_at(step)
        x, y = train.batch(a["batch"]); x, y = x.to(dev), y.to(dev)
        logits = run(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=0)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        step += 1; tok += int((y != 0).sum())
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.3f} lr {lr_at(step):.2e} {tok / (time.time() - t0):.0f} tok/s", flush=True)
        if step % a["eval"] == 0 or step == a["steps"]:
            model.eval(); vl, n = 0.0, 0
            with torch.no_grad():
                for x, y in vb:
                    lg = run(x.to(dev)); y = y.to(dev)
                    l = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1), ignore_index=0, reduction="sum")
                    vl += l.item(); n += int((y != 0).sum())
            vl /= n
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": min(best, vl), "cfg": model.cfg, "args": a}
            torch.save(ck, os.path.join(a["out"], "last.pt"))
            if vl < best:
                best = vl; torch.save({"model": model.state_dict(), "cfg": model.cfg, "step": step, "val": vl}, os.path.join(a["out"], "best.pt"))
            print(f"== step {step} held-out loss {vl:.3f} (best {best:.3f})", flush=True)

if __name__ == "__main__":
    main()
