# -*- coding: utf-8 -*-
"""Heavy-sample kernels: a Student-t mixture blended into a flow's density.

A flow ``q`` becomes ``(1 - f) q + f K``, with ``K`` an equal-weight mixture
of multivariate Student-t densities. :class:`HeavyKernel` is the kernel;
:class:`KernelMixFlow` blends it into a plain flow in the flow's own input
coordinates; a :class:`~nessai.flowmodel.group_mixture.DiscreteGroupMixtureFlowWrapper`
blends it into its base flow (``heavy_kernel``), in the standardised frame
its base flow sees, so the group's branches, sign patterns and domain
carry the kernel to every image of the point exactly as they carry the
base flow.
"""

import math

import numpy as np
import torch


class HeavyKernel:
    """Equal-weight mixture of Student-t kernels (``nu`` degrees of freedom),
    centres ``[m, d]`` and lower Cholesky factors of their scale matrices
    ``[m, d, d]``, blended with weight ``fraction``. Plain numpy state (it
    pickles with the proposal); torch tensors are made per call."""

    def __init__(self, centres, chols, nu, fraction):
        self.centres = np.asarray(centres, dtype=float)
        self.chols = np.asarray(chols, dtype=float)
        self.nu = float(nu)
        self.fraction = float(fraction)

    def _t(self, ref):
        return (
            torch.as_tensor(self.centres, dtype=ref.dtype, device=ref.device),
            torch.as_tensor(self.chols, dtype=ref.dtype, device=ref.device),
        )

    def log_prob(self, u):
        c, L = self._t(u)
        m, d = c.shape
        nu = self.nu
        const = (
            math.lgamma((nu + d) / 2)
            - math.lgamma(nu / 2)
            - 0.5 * d * math.log(nu * math.pi)
        )
        delta = (u.unsqueeze(0) - c.unsqueeze(1)).transpose(1, 2)  # m, d, n
        sol = torch.linalg.solve_triangular(L, delta, upper=False)
        maha = (sol**2).sum(dim=1)  # m, n
        log_det = torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(dim=1)
        lp = const - log_det[:, None] - 0.5 * (nu + d) * torch.log1p(maha / nu)
        return torch.logsumexp(lp, dim=0) - math.log(m)

    def sample(self, n, ref):
        c, L = self._t(ref)
        m, d = c.shape
        which = torch.randint(m, (n,), device=ref.device)
        z = torch.randn(n, d, dtype=ref.dtype, device=ref.device)
        g = torch.distributions.Chi2(
            torch.tensor(self.nu, dtype=ref.dtype, device=ref.device)
        ).sample((n,))
        z = z / torch.sqrt(g / self.nu)[:, None]
        return c[which] + torch.einsum("nij,nj->ni", L[which], z)

    def mix_log_prob(self, u, flow_log_prob):
        f = self.fraction
        return torch.logaddexp(
            math.log1p(-f) + flow_log_prob, math.log(f) + self.log_prob(u)
        )

    def replace_rows(self, u):
        """``u`` with a binomial fraction of its rows redrawn from the
        kernel: draws of the blended density when ``u`` are the flow's."""
        n = u.shape[0]
        if not n:
            return u
        pick = torch.rand(n, device=u.device) < self.fraction
        k = int(pick.sum())
        if not k:
            return u
        u = u.clone()
        u[pick] = self.sample(k, u)
        return u


class KernelMixFlow(torch.nn.Module):
    """A plain flow (``log_prob``, ``inverse`` with ``log_j`` such that
    ``base_distribution_log_prob(z) - log_j == log_prob(x)``) blended with a
    :class:`HeavyKernel` in its input coordinates."""

    def __init__(self, flow, kernel):
        super().__init__()
        self.flow = flow
        self.heavy_kernel = kernel

    def __getattr__(self, name):
        # the flow's other attributes (device, base distribution, ...)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("flow"), name)

    def log_prob(self, x, context=None):
        return self.heavy_kernel.mix_log_prob(
            x, self.flow.log_prob(x, context=context)
        )

    def base_distribution_log_prob(self, z, context=None):
        return self.flow.base_distribution_log_prob(z, context=context)

    def sample_latent_distribution(self, n, context=None):
        return self.flow.sample_latent_distribution(n, context=context)

    def forward(self, x, context=None):
        return self.flow.forward(x, context=context)

    def inverse(self, z, context=None):
        x, _ = self.flow.inverse(z, context=context)
        x = self.heavy_kernel.replace_rows(x)
        log_j = self.base_distribution_log_prob(z, context=context) - (
            self.log_prob(x, context=context)
        )
        return x, log_j

    def sample_and_log_prob(self, n, context=None):
        x, _ = self.flow.sample_and_log_prob(n, context=context)
        x = self.heavy_kernel.replace_rows(x)
        return x, self.log_prob(x, context=context)

    def sample(self, n, context=None):
        return self.sample_and_log_prob(n, context=context)[0]
