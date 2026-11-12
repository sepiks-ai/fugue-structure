"""A small decoder-only transformer over the piece tokens (src/fugue/tx/tokens.py): pre-norm blocks,
rotary positions (so it can be run past the length it was trained on), tied input and output embeddings."""
import math, torch, torch.nn as nn, torch.nn.functional as F

def rotary(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)

class Block(nn.Module):
    def __init__(s, d, h, drop):
        super().__init__()
        s.h = h; s.n1 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False)
        s.n2 = nn.LayerNorm(d); s.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)); s.drop = drop
    def forward(s, x, cos, sin):
        B, T, D = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=s.drop if s.training else 0.0)
        x = x + F.dropout(s.o(a.transpose(1, 2).reshape(B, T, D)), s.drop, s.training)
        return x + F.dropout(s.ff(s.n2(x)), s.drop, s.training)

class Composer(nn.Module):
    def __init__(s, vocab, d=384, layers=8, heads=6, drop=0.1, aux=False):
        super().__init__()
        s.cfg = dict(vocab=vocab, d=d, layers=layers, heads=heads, drop=drop, aux=aux)
        s.emb = nn.Embedding(vocab, d)
        # absolute pitch of the note a token puts down (0 for other tokens), added to its embedding; see tokens.aux_of
        s.pitch_emb = nn.Embedding(128, d) if aux else None
        s.blocks = nn.ModuleList(Block(d, heads, drop) for _ in range(layers))
        s.norm = nn.LayerNorm(d); s.hd = d // heads
        s.apply(s._init)
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, 0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
    def forward(s, idx, aux=None, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device, dtype=torch.float32)
        inv = 1.0 / (10000 ** (torch.arange(0, s.hd, 2, device=idx.device, dtype=torch.float32) / s.hd))
        ang = pos[:, None] * inv[None]; cos, sin = ang.cos(), ang.sin()
        x = s.emb(idx)
        if s.pitch_emb is not None and aux is not None: x = x + s.pitch_emb(aux)
        x = F.dropout(x, s.cfg["drop"], s.training)
        for b in s.blocks: x = b(x, cos, sin)
        logits = s.norm(x) @ s.emb.weight.T
        if targets is None: return logits
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
