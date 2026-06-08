import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    return (1 - alphas_bar[1:] / alphas_bar[:-1]).clamp(0.0001, 0.9999)


class DiffusionScheduler:
    def __init__(self, T: int, device: torch.device):
        self.T = T
        betas = cosine_beta_schedule(T).to(device)
        alphas = 1.0 - betas
        self.alphas_bar = torch.cumprod(alphas, dim=0)
        self.sqrt_ab = self.alphas_bar.sqrt()
        self.sqrt_1mab = (1.0 - self.alphas_bar).sqrt()

    def q_sample(self, x0, t):
        eps = torch.randn_like(x0)
        ab = self.sqrt_ab[t].view(-1, 1, 1, 1)
        s1m = self.sqrt_1mab[t].view(-1, 1, 1, 1)
        return ab * x0 + s1m * eps, eps


class SinusoidalPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return self.proj(torch.cat([args.sin(), args.cos()], dim=-1))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_mlp = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch * 2))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.t_mlp(t_emb).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[..., None, None]) + shift[..., None, None]
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.norm(x).view(b, c, -1).permute(0, 2, 1)
        o, _ = self.attn(q, q, q)
        return x + o.permute(0, 2, 1).view(b, c, h, w)


class MiniUNet(nn.Module):
    def __init__(self, base_ch=64, depth=4, T=100):
        super().__init__()
        t_dim = base_ch * 4
        self.t_emb = SinusoidalPE(base_ch)
        ch = [base_ch * (2**i) for i in range(depth)]
        self.enc_in = nn.Conv2d(4, ch[0], 3, padding=1)
        self.enc_res = nn.ModuleList()
        self.enc_down = nn.ModuleList()
        for i in range(depth - 1):
            self.enc_res.append(ResBlock(ch[i], ch[i], t_dim))
            self.enc_down.append(nn.Conv2d(ch[i], ch[i + 1], 4, stride=2, padding=1))
        self.mid_res1 = ResBlock(ch[-1], ch[-1], t_dim)
        self.mid_attn = AttentionBlock(ch[-1])
        self.mid_res2 = ResBlock(ch[-1], ch[-1], t_dim)
        self.dec_up = nn.ModuleList()
        self.dec_res = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.dec_up.append(nn.ConvTranspose2d(ch[i], ch[i - 1], 4, stride=2, padding=1))
            self.dec_res.append(ResBlock(ch[i - 1] * 2, ch[i - 1], t_dim))
        self.out_norm = nn.GroupNorm(8, ch[0])
        self.out_conv = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x, t, cond):
        t_emb = self.t_emb(t)
        h = self.enc_in(torch.cat([x, cond], dim=1))
        skips = []
        for res, down in zip(self.enc_res, self.enc_down):
            h = res(h, t_emb)
            skips.append(h)
            h = down(h)
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)
        for up, res, skip in zip(self.dec_up, self.dec_res, reversed(skips)):
            h = up(h)
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = res(torch.cat([h, skip], dim=1), t_emb)
        return self.out_conv(F.silu(self.out_norm(h)))
