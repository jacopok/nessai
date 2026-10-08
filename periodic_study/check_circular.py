"""Sanity checks of nessai.flows.circular: inverse, log-det, normalisation."""
import math
import numpy as np
import torch
from nessai.flows.circular import CircularNeuralSplineFlow, wrap

import sys
SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
NG = int(sys.argv[2]) if len(sys.argv) > 2 else 400
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

for real_transform in ["spline", "affine"]:
    for shift in ["angle", "real", None]:
        flow = CircularNeuralSplineFlow(4, 16, 4, 1, circular_features=[0, 2],
                                        real_transform=real_transform,
                                        circular_shift=shift, mask_seed=1,
                                        linear_transform="lu",
                                        batch_norm_between_layers=True)
        # randomise the conditioners so the transforms are far from identity
        for p in flow.parameters():
            p.data += SCALE * torch.randn_like(p)
        x = torch.randn(512, 4)
        x[:, [0, 2]] = math.pi * (2 * torch.rand(512, 2) - 1)
        # batch-norm running statistics start at zero variance: warm them
        with torch.no_grad():
            for _ in range(50):
                flow.forward(x)
        flow.eval()
        z, ld = flow.forward(x)
        xr, ldi = flow.inverse(z)
        err = (wrap(xr - x)).abs().max().item()
        # log-det via autograd (wrap has unit derivative a.e.)
        xs = x[:16].clone()
        jac = torch.autograd.functional.jacobian(lambda v: flow.forward(v)[0].sum(0), xs)
        jd = torch.stack([torch.linalg.slogdet(jac[:, i, :])[1] for i in range(16)])
        lderr = (jd - ld[:16]).abs().max().item()
        print(f"{real_transform:6s} {str(shift):5s} inv err {err:.2e}  "
              f"ld+ldi {((ld + ldi).abs().max().item()):.2e}  ld vs autograd {lderr:.2e}  "
              f"latent circ range [{z[:, [0,2]].min():.3f}, {z[:, [0,2]].max():.3f}]")

# normalisation on a pure torus (2 circular features), grid integral
flow = CircularNeuralSplineFlow(2, 16, 4, 1, circular_features=[0, 1], mask_seed=0)
for p in flow.parameters():
    p.data += SCALE * torch.randn_like(p)
flow.eval()
g = torch.linspace(-math.pi, math.pi, NG + 1)[:-1] + math.pi / NG
G = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
with torch.no_grad():
    lp = flow.log_prob(G)
print("torus integral", (lp.exp().sum() * (2 * math.pi / NG) ** 2).item())
# seam continuity: density just either side of -pi/pi
e = 1e-6
a = torch.tensor([[math.pi - e, 0.3], [-math.pi + e, 0.3], [0.3, math.pi - e], [0.3, -math.pi + e]])
with torch.no_grad():
    print("seam log densities", flow.log_prob(a).numpy())
# 1D circle + 1 real: integral
flow = CircularNeuralSplineFlow(2, 16, 4, 1, circular_features=[0], mask_seed=0)
for p in flow.parameters():
    p.data += SCALE * torch.randn_like(p)
flow.eval()
r = torch.linspace(-12, 12, 1201)
G = torch.stack(torch.meshgrid(g, r, indexing="ij"), -1).reshape(-1, 2)
with torch.no_grad():
    lp = flow.log_prob(G)
print("circle x line integral", (lp.exp().sum() * (2 * math.pi / NG) * (24 / 1200)).item())
x, lq = flow.sample_and_log_prob(1000)
print("sample_and_log_prob vs log_prob", (lq - flow.log_prob(x)).abs().max().item())
