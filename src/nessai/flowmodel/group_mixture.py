"""
Discrete group-mixture flow extension for nessai.
"""

import logging
import math
import os

import numpy as np
import torch
from torch.distributions import Categorical

from ..flows.base import BaseFlow
from ..flows.utils import configure_model
from ..livepoint import empty_structured_array, live_points_to_array
from ..proposal import FlowProposal
from .base import FlowModel

logger = logging.getLogger(__name__)


class CoordinateBridge:
    """Map between the flow's ``prime`` coords and the user's ``physical`` coords.

    The group-mixture flow applies the *conjugated* action ``g_hat = T^-1 . g
    . T`` in prime space, where ``T`` is the prime->physical map
    (``physical = T(prime)``). ``L(prime) = log|det dphysical/dprime|`` at a
    prime point; the log-determinant of the conjugated map ``prime_in ->
    prime_out`` is then ``L(prime_in) - L(prime_out)`` (the physical action
    ``g`` is assumed measure preserving).

    Subclasses implement :meth:`to_physical` / :meth:`to_prime`. For an affine
    ``T`` (:class:`AffineBridge`) ``L`` is constant and every conjugation
    log-determinant vanishes; the general :class:`ReparamBridge` round-trips
    through a nessai reparameterisation in numpy and carries the exact ``L``.
    """

    is_affine = False
    #: prime-space dimension differs from physical (augmented reparams)
    dimension_changing = False

    def to_physical(self, prime):
        """``prime [N, d_prime] -> (physical [N, d_phys], L [N], aux)``."""
        raise NotImplementedError

    def to_prime(self, physical, aux=None):
        """``physical [N, d_phys] -> (prime [N, d_prime], L [N])``."""
        raise NotImplementedError


class AffineBridge(CoordinateBridge):
    """Diagonal affine map ``physical = prime * scale + shift``."""

    is_affine = True

    def __init__(self, scale, shift):
        self.scale = scale
        self.shift = shift
        self._logdet = torch.log(scale.abs()).sum()

    def to_physical(self, prime):
        phys = prime * self.scale + self.shift
        L = self._logdet.expand(prime.shape[0])
        return phys, L, None

    def to_prime(self, physical, aux=None):
        prime = (physical - self.shift) / self.scale
        L = self._logdet.expand(physical.shape[0])
        return prime, L


class ReparamBridge(CoordinateBridge):
    """General bridge round-tripping through a nessai reparameterisation.

    ``forward_fn`` / ``inverse_fn`` are numpy callables taking a structured
    livepoint array and returning ``(structured_array, log_J)`` where ``log_J``
    is the reparameterisation's exact per-sample log-Jacobian for that
    direction. ``rescale`` maps physical->prime, ``inverse_rescale`` maps
    prime->physical. The reparameterisation ``log_J`` for prime->physical is
    exactly ``L``; the physical->prime pass returns ``-L`` (up to numerical
    error), so both directions are reconciled here.

    The *physical* side is carried in ``physical_dtype`` (float64 by default),
    not in the flow's ``dtype``. Physical parameters are not standardised, so
    they can be huge next to the spread the posterior actually resolves -- a
    GW ``geocent_time`` is a GPS time of order ``1.2e9`` with a prior only
    ``0.2`` s wide, and the float32 spacing there is ``128`` s. Storing the
    physical point in float32 would snap every sample onto the same value and
    the prime->physical->prime round trip would destroy that coordinate, so
    ``sample_and_log_prob`` and :meth:`log_prob` would then score different
    points. Only the prime coordinates the flow itself sees are cast to
    ``dtype``.
    """

    def __init__(
        self,
        prime_names,
        physical_names,
        rescale_fn,
        inverse_rescale_fn,
        dtype,
        device,
        physical_dtype=torch.float64,
    ):
        self.prime_names = list(prime_names)
        self.physical_names = list(physical_names)
        self.rescale_fn = rescale_fn
        self.inverse_rescale_fn = inverse_rescale_fn
        self.dtype = dtype
        self.physical_dtype = physical_dtype
        self.device = device
        self.dimension_changing = len(self.prime_names) != len(
            self.physical_names
        )

    def _structured(self, array, names):
        out = empty_structured_array(len(array), names=list(names))
        for i, name in enumerate(names):
            out[name] = array[:, i]
        return out

    def _to_tensor(self, array, dtype=None):
        return torch.as_tensor(
            array,
            dtype=self.dtype if dtype is None else dtype,
            device=self.device,
        )

    def to_physical(self, prime):
        prime_np = prime.detach().cpu().numpy().astype(float)
        struct = self._structured(prime_np, self.prime_names)
        phys_struct, log_j = self.inverse_rescale_fn(struct)
        phys = live_points_to_array(
            phys_struct, self.physical_names, copy=True
        )
        L = self._to_tensor(np.asarray(log_j, dtype=float))
        aux = None
        if self.dimension_changing:
            aux_names = [
                n for n in self.prime_names if n not in self.physical_names
            ]
            aux = self._to_tensor(
                live_points_to_array(phys_struct, aux_names, copy=True),
                dtype=self.physical_dtype,
            )
        return self._to_tensor(phys, dtype=self.physical_dtype), L, aux

    def to_prime(self, physical, aux=None):
        phys_np = physical.detach().cpu().numpy().astype(float)
        struct = self._structured(phys_np, self.physical_names)
        prime_struct, log_j = self.rescale_fn(struct)
        prime = live_points_to_array(
            prime_struct, self.prime_names, copy=True
        )
        # rescale returns log|det dprime/dphysical| = -L.
        L = -self._to_tensor(np.asarray(log_j, dtype=float))
        prime_t = self._to_tensor(prime)
        if self.dimension_changing and aux is not None:
            aux_names = [
                n for n in self.prime_names if n not in self.physical_names
            ]
            idx = [self.prime_names.index(n) for n in aux_names]
            prime_t[:, idx] = aux.to(prime_t)
        return prime_t, L


class DiscreteGroupMixtureFlowWrapper(BaseFlow):
    """Wrap a base flow with a discrete group-mixture transformation.

    Computes ``log p(x) = logsumexp_g [log p_base(g^-1 x) + log pi_g]``. The
    mixture weights ``pi_g`` are not trained by gradient descent: the
    fundamental-domain mask assigns each point to exactly one group element,
    so the maximum-likelihood weights are the assigned-point fractions, set
    in closed form by :meth:`update_mixture_weights`.

    The group action and fundamental-domain predicate are written by the user
    in *physical* coordinates. A :class:`CoordinateBridge` (installed by the
    proposal via :meth:`set_coordinate_bridge`) maps between physical and the
    flow's *prime* coordinates and supplies the conjugation log-Jacobian, so
    non-affine reparameterisations are handled exactly. Alternatively the user
    can pass ``prime_space_action`` / ``prime_space_in_domain`` to work
    directly in prime coordinates and bypass the bridge entirely.

    The base flow has full support, so a generative draw ``canon`` can land
    outside the fundamental domain. By default (``truncate_base_to_domain``)
    :meth:`sample_and_log_prob` rejects those draws and renormalises by the
    tracked base mass ``Z`` inside the domain (``_log_domain_mass``), so the
    single-branch density is exact and the importance weights carry no leakage
    bias. Set ``truncate_base_to_domain=False`` to instead keep every draw and
    score it with the leaky full-mixture fallback in
    :meth:`_mixture_log_prob_from_canonical`. With no fundamental-domain
    predicate truncation is unavailable and the fallback is always used.
    """

    def __init__(
        self,
        base_flow,
        num_features,
        group_action_fn,
        group_size,
        param_names=None,
        in_fundamental_domain=None,
        prime_space_action=None,
        prime_space_in_domain=None,
        min_canon_std=1e-2,
        truncate_base_to_domain=True,
        reflect_parameters=None,
        canonical_transform=None,
        mode_factor_sizes=None,
    ):
        super().__init__()
        self.base_flow = base_flow
        self.num_features = num_features
        self.group_size = group_size
        self.group_action_fn = group_action_fn
        self.in_fundamental_domain = in_fundamental_domain
        self.prime_space_action = prime_space_action
        self.prime_space_in_domain = prime_space_in_domain
        self.truncate_base_to_domain = truncate_base_to_domain
        # Truncation needs a fundamental-domain predicate; without one it is
        # silently a no-op (the leaky single-branch shortcut is then the only
        # option). ``_truncate`` is the effective switch used at runtime.
        _has_predicate = (
            prime_space_in_domain is not None
            if prime_space_action is not None
            else in_fundamental_domain is not None
        )
        self._truncate = bool(truncate_base_to_domain and _has_predicate)
        if truncate_base_to_domain and not self._truncate:
            logger.info(
                "truncate_base_to_domain is set but no fundamental-domain "
                "predicate was given; base-flow truncation is disabled and "
                "sample_and_log_prob falls back to the leaky single-branch "
                "shortcut."
            )
        self.param_names = param_names or [
            f"p_{i}" for i in range(num_features)
        ]

        # Boundary reflection: canonical coordinates that sit against a hard
        # fundamental-domain wall at 0 (e.g. the ``x, y, z >= 0`` octant faces
        # of a rotated sky decomposition). The base flow then models the
        # sign-symmetric extension ``q0_sym(u) = sum_s q0(s . u)`` over the
        # 2**m sign patterns of these dims, so it never has to represent the
        # wall cliff; a generative draw is folded back with ``abs``.
        self.reflect_parameters = list(reflect_parameters or [])
        if self.reflect_parameters and prime_space_action is None:
            logger.warning(
                "reflect_parameters is only supported on the prime-space "
                "action path; ignoring %s.",
                self.reflect_parameters,
            )
            self.reflect_parameters = []
        self._reflect_idx = torch.empty(0, dtype=torch.long)
        self.register_buffer("_sign_patterns", None)
        self._configure_reflection()

        # Optional fixed analytic bijection between the wrapper's *canonical*
        # prime coordinates and the coordinates the base flow actually models
        # (``t``). Dimension-preserving. Used to reshape a hard canonical
        # geometry -- e.g. a uniform sky octant with a sharp prior edge -- into
        # something closer to the base flow's Gaussian latent before the
        # per-element standardisation is applied. Contract:
        #   forward(canon) -> (t,    log|det dt/dcanon|)   [N, d], [N]
        #   inverse(t)     -> (canon, log|det dcanon/dt|)  [N, d], [N]
        # The transform sits *inside* the canonical fundamental domain, so the
        # group action, domain predicate and conjugation Jacobians are all
        # unaffected; only ``prime_space_action`` runs are supported.
        self._canonical_transform = canonical_transform
        if canonical_transform is not None and prime_space_action is None:
            raise ValueError(
                "canonical_transform is only supported on the "
                "prime_space_action path."
            )
        if canonical_transform is not None and hasattr(
            canonical_transform, "bind"
        ):
            canonical_transform.bind(self.param_names)

        # Coordinate bridge; defaults to the identity affine map so the
        # wrapper is usable without a proposal (e.g. in unit tests).
        self._bridge = AffineBridge(
            torch.ones(num_features), torch.zeros(num_features)
        )
        # Affine scale/shift kept as buffers for backward compatibility and
        # for the analytic canonical-buffer re-expression on the affine path.
        self.register_buffer("_prime_scale", torch.ones(num_features))
        self.register_buffer("_prime_shift", torch.zeros(num_features))

        # Mixture weights, set in closed form by ``update_mixture_weights``.
        self.register_buffer(
            "weights", torch.full((group_size,), 1.0 / group_size)
        )

        # Optional factorisation of the group into commuting cyclic factors
        # (``mode_factor_sizes = [s_0, ..., s_{F-1}]``, ``prod = group_size``),
        # with the mode index a little-endian mixed-radix code:
        # ``factor_f(g) = (g // prod(s_{<f})) % s_f``. When present,
        # :meth:`update_mixture_weights` estimates the ``F`` marginal
        # distributions over the factors independently (each pooling counts
        # over every other factor) and sets ``pi_g`` to their product. A
        # single starved joint mode then keeps a non-zero weight as long as
        # its per-factor marginals are populated, which makes the estimator
        # far more robust to transient mode collapse than the flat
        # per-mode count. Residual bias from non-independent factors (only
        # the approximate phase symmetry is a plausible offender, and roughly
        # independently of the sky/reflection factors) is corrected by the
        # downstream importance reweighting.
        if mode_factor_sizes:
            mode_factor_sizes = [int(s) for s in mode_factor_sizes]
            prod = 1
            for s in mode_factor_sizes:
                prod *= s
            if prod != group_size:
                raise ValueError(
                    f"mode_factor_sizes {mode_factor_sizes} multiply to "
                    f"{prod}, not group_size={group_size}."
                )
        else:
            mode_factor_sizes = None
        self.mode_factor_sizes = mode_factor_sizes
        if mode_factor_sizes is not None:
            g = torch.arange(group_size)
            cols, stride = [], 1
            for s in mode_factor_sizes:
                cols.append(torch.div(g, stride, rounding_mode="floor") % s)
                stride *= s
            self.register_buffer(
                "_mode_factor_index", torch.stack(cols, dim=1)
            )
            self.register_buffer(
                "_factor_empty_rounds",
                torch.zeros(sum(mode_factor_sizes), dtype=torch.long),
            )
            self._factor_offsets = [0]
            for s in mode_factor_sizes:
                self._factor_offsets.append(self._factor_offsets[-1] + s)
        else:
            self.register_buffer("_mode_factor_index", None)
            self.register_buffer("_factor_empty_rounds", None)
            self._factor_offsets = None

        # Per-element standardisation of the canonical coordinates seen by
        # the base flow, refreshed by ``update_base_standardisation``. It
        # rescales every mode's surviving region to a common size so a
        # single ``q0`` shape fits them all.
        self.register_buffer(
            "_canon_mean", torch.zeros(group_size, num_features)
        )
        self.register_buffer(
            "_canon_std", torch.ones(group_size, num_features)
        )
        self.register_buffer(
            "_canon_seen", torch.zeros(group_size, dtype=torch.bool)
        )
        # Consecutive rounds each element has been empty; an element empty
        # for ``_weight_empty_patience`` rounds is dropped (weight 0) until
        # points return to it.
        self.register_buffer(
            "_empty_rounds", torch.zeros(group_size, dtype=torch.long)
        )
        self._min_std_count = 16
        self._canon_ema = 0.3
        self._weight_empty_patience = 3
        # Floor for the per-element canonical std. The canonical coordinates
        # live in the flow's ~unit-scaled prime frame, so a per-element std
        # far below 1 means that dimension carries almost no information for
        # that mode (e.g. a parameter the run effectively fixes). Rescaling it
        # to O(1) anyway divides by a near-zero number: ``_canon_log_det``
        # blows up and ``sample_and_log_prob`` (which draws that dim with
        # width ``_canon_std``) stops agreeing with ``log_prob`` (which sees
        # the true, much narrower spread). Flooring keeps both paths finite
        # and consistent; the residual sub-floor variance is modelled by the
        # base flow instead.
        #
        # The default (1e-2) suits a ~unit-scaled prime frame. A run that
        # z-scores a parameter far tighter than the prior resolves it (e.g. a
        # GW ``geocent_time`` pinned to ~1e-4 s inside a 0.2 s prior) has a
        # true prime spread orders of magnitude below the floor: the base flow
        # is then asked to model a near-delta in that coordinate and
        # ``sample_and_log_prob`` comes out over-dispersed there. Lower
        # ``min_canon_std`` in that case; ``update_base_standardisation`` warns
        # once when the floor actually binds.
        self._min_canon_std = min_canon_std
        self._warned_canon_clamp = False
        # Fraction of the last batch whose generative draw leaked out of the
        # canonical fundamental domain and so fell back from the single-branch
        # shortcut to the full mixture ``log_prob`` in
        # ``_mixture_log_prob_from_canonical``. Split into the two disjoint
        # causes: ``_domain`` -- ``canon`` left the fundamental domain (the
        # base flow put mass outside the box it tiles, often outside the prior
        # altogether); ``_roundtrip`` -- ``canon`` is in-domain but the inverse
        # action of its branch does not map ``x`` back onto it (a non-injective
        # action that clamps / saturates outside a box, e.g. raw angles through
        # ``asin(clamp(...))``). ``_last_leakage_fraction`` is their union.
        self._last_leakage_fraction = 0.0
        self._last_leakage_fraction_domain = 0.0
        self._last_leakage_fraction_roundtrip = 0.0

        # Diagnostics for the factorised weight estimator: the per-factor
        # marginal distributions and raw assignment counts from the last
        # ``update_mixture_weights`` call (``None`` on the flat path). Consumed
        # by ``GroupFlowProposalMixin._log_group_weight_entropy``.
        self._last_factor_marginals = None
        self._last_factor_counts = None

        # ``truncate_base_to_domain``: reject a generative draw whose ``canon``
        # falls outside the fundamental domain instead of scoring it with the
        # leaky single-branch shortcut. The base density then becomes ``q0``
        # restricted to the domain and renormalised by its mass there,
        # ``Z = P_{q0}(canon in D)``; every other branch contributes *exactly*
        # zero, so the single-branch density is exact up to the ``-log Z``
        # offset. ``Z`` has no closed form -- it is tracked as an EMA of the
        # per-batch acceptance rate of the rejection loop and applied in
        # :meth:`log_prob` / :meth:`_mixture_log_prob_from_canonical`.
        self.register_buffer("_log_domain_mass", torch.zeros(()))
        self._domain_mass_seen = False
        self._domain_mass_ema = 0.3

    @property
    def uses_prime_space_action(self):
        return self.prime_space_action is not None

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Forward compatibility: a checkpoint written before the factorised
        # weight estimator predates ``_mode_factor_index`` /
        # ``_factor_empty_rounds``. Fill any missing buffer with its current
        # (config-derived) default so ``strict=True`` resume still works.
        sd = dict(state_dict)
        for name, val in super().state_dict().items():
            sd.setdefault(name, val)
        return super().load_state_dict(sd, strict=strict, assign=assign)

    def _configure_reflection(self):
        """(Re)build the reflect-dim indices and sign patterns from
        ``reflect_parameters`` against the current ``param_names``."""
        idx = [
            self.param_names.index(p)
            for p in self.reflect_parameters
            if p in self.param_names
        ]
        self._reflect_idx = torch.tensor(idx, dtype=torch.long)
        if idx:
            m = len(idx)
            grid = torch.cartesian_prod(
                *[torch.tensor([1.0, -1.0])] * m
            ).reshape(2**m, m)
            signs = torch.ones(2**m, self.num_features)
            signs[:, self._reflect_idx] = grid
            ref = self._sign_patterns
            if isinstance(ref, torch.Tensor):
                signs = signs.to(ref.device, ref.dtype)
            self._sign_patterns = signs
        else:
            self._sign_patterns = None

    def set_param_names(self, names):
        """Rebind the prime-parameter names (e.g. once the proposal knows the
        reparameterisation's true order) and refresh reflection indices."""
        self.param_names = list(names)
        self._configure_reflection()
        if self._canonical_transform is not None and hasattr(
            self._canonical_transform, "bind"
        ):
            self._canonical_transform.bind(self.param_names)

    def set_coordinate_bridge(self, bridge):
        """Install a :class:`CoordinateBridge`.

        The canonical standardisation buffers live in prime coordinates. An
        affine frame change re-expresses them analytically; a non-affine
        change cannot, so the buffers are kept and left for
        :meth:`update_base_standardisation` to re-adapt (a large shift resets
        ``_canon_seen`` so modes re-bootstrap).
        """
        old = self._bridge
        if isinstance(bridge, AffineBridge):
            self.set_affine_maps(bridge.scale, bridge.shift)
            return
        if isinstance(old, AffineBridge) and bool(self._canon_seen.any()):
            # Leaving the affine fast path: probe the frame shift on the
            # stored canonical means and reset modes that moved a lot.
            with torch.no_grad():
                phys_old, _, _ = old.to_physical(self._canon_mean)
                prime_new, _ = bridge.to_prime(phys_old)
                shift = (prime_new - self._canon_mean).abs()
                moved = (shift > 2.0 * self._canon_std).any(dim=-1)
                self._canon_seen[moved] = False
        self._bridge = bridge

    def set_affine_maps(self, scale, shift):
        """Set the prime->physical affine map.

        The canonical standardisation buffers are stored in prime
        coordinates, so when the prime frame moves they are re-expressed in
        the new frame.
        """
        scale = scale.to(self._prime_scale)
        shift = shift.to(self._prime_shift)
        if bool(self._canon_seen.any()):
            ratio = self._prime_scale / scale
            offset = (self._prime_shift - shift) / scale
            self._canon_mean.mul_(ratio).add_(offset)
            self._canon_std.mul_(ratio.abs())
        self._prime_scale.copy_(scale)
        self._prime_shift.copy_(shift)
        self._bridge = AffineBridge(scale.clone(), shift.clone())

    def _to_physical(self, z):
        phys, _, _ = self._bridge.to_physical(z)
        return phys

    def _to_prime(self, x):
        prime, _ = self._bridge.to_prime(x)
        return prime

    def _to_base(self, canon):
        """Canonical prime coords -> base-flow coords ``t`` and ``log|dt/dcanon|``."""
        if self._canonical_transform is None:
            return canon, canon.new_zeros(canon.shape[0])
        return self._canonical_transform.forward(canon)

    def _from_base(self, t):
        """Base-flow coords ``t`` -> canonical prime coords and ``log|dcanon/dt|``."""
        if self._canonical_transform is None:
            return t, t.new_zeros(t.shape[0])
        return self._canonical_transform.inverse(t)

    def _standardise(self, canon, modes):
        return (canon - self._canon_mean[modes]) / self._canon_std[modes]

    def _destandardise(self, u, modes):
        return u * self._canon_std[modes] + self._canon_mean[modes]

    def _canon_log_det(self, modes):
        return -torch.log(self._canon_std[modes]).sum(-1)

    def _fold_reflect(self, canon):
        """Fold canonical coords back to the positive side of each reflect wall.

        ``update_base_standardisation`` pins ``_canon_mean = 0`` on the reflect
        dims, so the wall at ``canon_i = 0`` is also the standardisation
        centre; ``abs`` is the correct fold.
        """
        if self._sign_patterns is None:
            return canon
        canon = canon.clone()
        idx = self._reflect_idx.to(canon.device)
        canon[:, idx] = canon[:, idx].abs()
        return canon

    def _base_log_prob(self, canon, modes, context=None):
        """``log q0`` of standardised ``canon``, symmetrised over reflect dims.

        Without reflect dims this is just ``base_flow.log_prob`` of the
        standardised point. With them it returns
        ``logsumexp_s base_flow.log_prob(s . standardise(canon))`` over the
        2**m sign patterns -- the density on the positive orthant whose
        sign-symmetric extension is the base flow (unit-normalised, no extra
        constant; the ``_canon_log_det`` term is added by the caller as usual).

        A ``canonical_transform`` (if set) is applied first: ``canon -> t`` with
        its ``log|det dt/dcanon|`` folded into the return value, and the
        standardisation / reflection then act on ``t``.
        """
        t, log_j = self._to_base(canon)
        u = self._standardise(t, modes)
        if self._sign_patterns is None:
            return self.base_flow.log_prob(u, context=context) + log_j
        p = self._sign_patterns.shape[0]
        n, f = u.shape
        signs = self._sign_patterns.to(u.dtype)
        u_rep = (u.unsqueeze(0) * signs.unsqueeze(1)).reshape(p * n, f)
        lp = self.base_flow.log_prob(u_rep, context=context).view(p, n)
        return torch.logsumexp(lp, dim=0) + log_j

    def _apply_group_action(self, z_flat, modes_flat, inverse):
        """Apply the conjugated action to prime points.

        Returns ``(mapped_prime [N, d], conj_logdet [N])`` where
        ``conj_logdet`` is the log-determinant of the prime-space map. On the
        affine path with a measure-preserving physical action it is zero.

        The physical action is assumed measure preserving by default. An action
        that is *not* (e.g. one written in the raw angles ``dec`` / ``theta_jn``
        rather than ``sin_dec`` / ``cos_theta_jn``) may instead return
        ``(point_dict, log_det)``, where ``log_det [N]`` is the log-determinant
        of the physical map it applied; it is threaded into ``conj_logdet``.
        """
        if self.uses_prime_space_action:
            point_dict = {
                name: z_flat[:, i]
                for i, name in enumerate(self.param_names)
            }
            out = self.prime_space_action(
                point_dict, modes_flat, inverse=inverse
            )
            if isinstance(out, tuple):
                mapped_dict, logdet = out
            else:
                mapped_dict, logdet = out, z_flat.new_zeros(z_flat.shape[0])
            mapped = torch.stack(
                [mapped_dict[name] for name in self.param_names], dim=-1
            )
            return mapped, logdet

        phys, L_in, aux = self._bridge.to_physical(z_flat)
        point_dict = {
            name: phys[:, i] for i, name in enumerate(self.physical_names)
        }
        out = self.group_action_fn(point_dict, modes_flat, inverse=inverse)
        if isinstance(out, tuple):
            mapped_dict, phys_logdet = out
        else:
            mapped_dict, phys_logdet = out, None
        phys_out = torch.stack(
            [mapped_dict[name] for name in self.physical_names], dim=-1
        )
        z_out, L_out = self._bridge.to_prime(phys_out, aux=aux)
        conj_logdet = L_in - L_out
        if phys_logdet is not None:
            conj_logdet = conj_logdet + torch.as_tensor(
                phys_logdet, dtype=conj_logdet.dtype, device=conj_logdet.device
            )
        return z_out, conj_logdet

    @property
    def physical_names(self):
        # For a dimension-changing bridge the physical names are the user's
        # ``param_names``; prime names carry extra auxiliaries.
        return self.param_names

    def _in_domain(self, z_flat):
        """Boolean mask: which rows of ``z_flat`` lie in the fundamental domain."""
        if self.uses_prime_space_action:
            point_dict = {
                name: z_flat[:, i]
                for i, name in enumerate(self.param_names)
            }
            return self.prime_space_in_domain(point_dict).to(torch.bool)
        phys, _, _ = self._bridge.to_physical(z_flat)
        point_dict = {
            name: phys[:, i] for i, name in enumerate(self.physical_names)
        }
        return self.in_fundamental_domain(point_dict).to(torch.bool)

    def _preimages(self, x):
        """All group pre-images ``g_k^-1 . x`` in prime coords.

        Returns ``(pre [K, B, d], conj_logdet [K, B])``.
        """
        k, b = self.group_size, x.shape[0]
        x_rep = x.unsqueeze(0).expand(k, b, -1).reshape(k * b, -1)
        modes = torch.arange(k, device=x.device).repeat_interleave(b)
        pre, logdet = self._apply_group_action(x_rep, modes, inverse=True)
        return pre.view(k, b, -1), logdet.view(k, b)

    def _assign_branch(self, x):
        """Geometric branch assignment (no base-flow evaluation).

        Returns ``(assigned [B], preimages [K, B, d], claimed [B])`` where
        ``assigned`` is the group element whose inverse maps ``x`` into the
        fundamental domain (0 for points no element claims).
        """
        pre, _ = self._preimages(x)
        if self.in_fundamental_domain is None and not self.uses_prime_space_action:
            return (
                torch.zeros(x.shape[0], dtype=torch.long, device=x.device),
                pre,
                torch.ones(x.shape[0], dtype=torch.bool, device=x.device),
            )
        k, b, d = pre.shape
        in_dom = self._in_domain(pre.reshape(k * b, d)).view(k, b)
        claimed = in_dom.any(dim=0)
        assigned = torch.where(claimed, in_dom.float().argmax(dim=0), 0)
        return assigned, pre, claimed

    @torch.no_grad()
    def update_base_standardisation(self, x, context=None):
        """Update the per-element canonical standardisation from assigned points.

        Each element's canonical points are standardised by their own
        mean/std, exponentially averaged across rounds. An element with too
        few points this round keeps its last value; an element never yet
        seen is bootstrapped from the pooled mean/std.
        """
        b = x.shape[0]
        assigned, pre, claimed = self._assign_branch(x)
        canon = pre[assigned, torch.arange(b, device=x.device)]
        # The base flow (hence the standardisation buffers and the reflect
        # walls) lives in the transformed frame ``t``, not raw ``canon``.
        canon, _ = self._to_base(canon)

        def _pin_reflect(mean, std, data):
            # Reflect dims are symmetrised about their wall at 0, so the
            # standardisation centre must be 0 and the scale the RMS about 0
            # (the second moment of the folded data), not the one-sided
            # mean/std.
            if self._sign_patterns is None:
                return mean, std
            idx = self._reflect_idx.to(data.device)
            rms = (
                data[:, idx].pow(2).mean(dim=0).clamp_min(
                    self._min_canon_std**2
                ).sqrt()
            )
            mean = mean.clone()
            std = std.clone()
            mean[idx] = 0.0
            std[idx] = rms
            return mean, std

        gmean = canon.mean(dim=0)
        raw_std = canon.std(dim=0)
        gstd = raw_std.clamp_min(self._min_canon_std)
        gmean, gstd = _pin_reflect(gmean, gstd, canon)
        clamped = raw_std < self._min_canon_std
        if bool(clamped.any()) and not self._warned_canon_clamp:
            names = [
                self.param_names[i]
                for i in clamped.nonzero(as_tuple=True)[0].tolist()
            ]
            logger.warning(
                "Group-mixture canonical std floored at %g for %s: the "
                "posterior is far narrower than the prime frame there, so the "
                "base flow must model a near-delta in that coordinate and "
                "sample_and_log_prob may be over-dispersed. Lower "
                "min_canon_std or use a tighter reparameterisation.",
                self._min_canon_std,
                names,
            )
            self._warned_canon_clamp = True
        beta = self._canon_ema
        for k in range(self.group_size):
            sel = claimed & (assigned == k)
            if int(sel.sum()) >= self._min_std_count:
                c = canon[sel]
                mk = c.mean(dim=0)
                sk = c.std(dim=0).clamp_min(self._min_canon_std)
                mk, sk = _pin_reflect(mk, sk, c)
                if self._canon_seen[k]:
                    self._canon_mean[k].mul_(1 - beta).add_(beta * mk)
                    self._canon_std[k].mul_(1 - beta).add_(beta * sk)
                else:
                    self._canon_mean[k].copy_(mk)
                    self._canon_std[k].copy_(sk)
                    self._canon_seen[k] = True
            elif not self._canon_seen[k]:
                self._canon_mean[k].copy_(gmean)
                self._canon_std[k].copy_(gstd)

    @torch.no_grad()
    def update_mixture_weights(self, x, context=None, smoothing=1.0):
        """EM M-step: set the mixture weights from the assigned-point counts.

        Each point is assigned to the single group element whose inverse
        action maps it into the fundamental domain. Without a group
        factorisation the maximum-likelihood weights are the per-mode
        assignment fractions; ``smoothing`` keeps a transiently-empty element
        in play and an element empty for ``_weight_empty_patience``
        consecutive rounds is dropped (weight 0) until points return to it.

        With ``mode_factor_sizes`` set the weights are instead the product of
        the ``F`` per-factor marginal distributions, each estimated from the
        counts of that factor's value pooled over every other factor (see
        :meth:`__init__`). The same smoothing / empty-patience rule then acts
        per factor *value*, so a starved joint mode is only zeroed when a
        whole marginal slice empties -- a much rarer event.
        """
        assigned, _, claimed = self._assign_branch(x)
        a = assigned[claimed]

        if self.mode_factor_sizes is None:
            counts = torch.bincount(a, minlength=self.group_size).to(
                self.weights
            )
            self._empty_rounds = torch.where(
                counts == 0,
                self._empty_rounds + 1,
                torch.zeros_like(self._empty_rounds),
            )
            active = self._empty_rounds < self._weight_empty_patience
            smoothed = torch.where(
                active, counts + smoothing, torch.zeros_like(counts)
            )
            self.weights.copy_(smoothed / smoothed.sum())
            return

        if a.numel() == 0:
            # No claimed points this round: keep the current weights rather
            # than dividing 0/0.
            return
        log_w = torch.zeros_like(self.weights)
        marginals, raw_counts = [], []
        for f, size in enumerate(self.mode_factor_sizes):
            off = self._factor_offsets[f]
            fac_of_mode = self._mode_factor_index[:, f]
            fcounts = torch.bincount(
                fac_of_mode[a], minlength=size
            ).to(self.weights)
            er = self._factor_empty_rounds[off : off + size]
            er = torch.where(fcounts == 0, er + 1, torch.zeros_like(er))
            self._factor_empty_rounds[off : off + size] = er
            active = er < self._weight_empty_patience
            smoothed = torch.where(
                active, fcounts + smoothing, torch.zeros_like(fcounts)
            )
            total = smoothed.sum()
            if total == 0:
                # Every value of this factor was dropped -- fall back to a
                # uniform marginal so the product stays well defined.
                p_f = torch.full_like(smoothed, 1.0 / size)
            else:
                p_f = smoothed / total
            log_w = log_w + torch.log(p_f[fac_of_mode])
            marginals.append(p_f.detach().cpu())
            raw_counts.append(fcounts.detach().cpu())
        self.weights.copy_(torch.softmax(log_w, dim=0))
        self._last_factor_marginals = marginals
        self._last_factor_counts = raw_counts
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Group-mixture factor marginals: %s",
                " | ".join(
                    f"f{f}[{'x'.join(map(str, self.mode_factor_sizes))}]="
                    + np.array2string(
                        m.numpy(), precision=3, separator=",",
                        suppress_small=True,
                    )
                    for f, m in enumerate(marginals)
                ),
            )

    def _branch_log_probs(self, x, context=None):
        """Return ``base_lp(g_k^-1 x) + log pi_k`` for every group element: ``[K, B]``.

        Branches whose pre-image is outside the fundamental domain are
        ``-inf`` and the base flow is not evaluated for them, so for a group
        that tiles the space this costs a single base-flow call on ``B``
        canonical points rather than ``K * B``. Each branch carries the
        conjugation log-Jacobian ``conj_logdet`` (zero on the affine path).
        """
        k, b = self.group_size, x.shape[0]
        pre, conj_logdet = self._preimages(x)
        log_pi = torch.log(self.weights).unsqueeze(1).expand(k, b)
        flat_pre = pre.reshape(k * b, -1)
        flat_modes = torch.arange(k, device=x.device).repeat_interleave(b)
        flat_conj = conj_logdet.reshape(k * b)

        if self.in_fundamental_domain is None and not self.uses_prime_space_action:
            base_lp = (
                self._base_log_prob(flat_pre, flat_modes, context=context)
                + self._canon_log_det(flat_modes)
                + flat_conj
            ).view(k, b)
            return base_lp + log_pi

        in_dom = self._in_domain(flat_pre).view(k, b)
        # Points no branch claims: evaluate all their branches (fallback).
        eval_mask = in_dom | (~in_dom.any(dim=0, keepdim=True))
        idx = eval_mask.reshape(-1).nonzero(as_tuple=True)[0]
        base_lp = flat_pre.new_full((k * b,), -float("inf"))
        if idx.numel():
            good = torch.isfinite(flat_pre[idx]).all(dim=-1)
            gi = idx[good]
            if gi.numel():
                base_lp[gi] = (
                    self._base_log_prob(
                        flat_pre[gi], flat_modes[gi], context=context
                    )
                    + self._canon_log_det(flat_modes[gi])
                    + flat_conj[gi]
                )
        return base_lp.view(k, b) + log_pi

    def _raw_log_prob(self, x, context=None):
        return torch.logsumexp(
            self._branch_log_probs(x, context=context), dim=0
        )

    def log_prob(self, x, context=None):
        lp = self._raw_log_prob(x, context=context)
        if self._truncate:
            lp = lp - self._log_domain_mass
        return lp

    def _update_domain_mass(self, n_kept, n_drawn):
        """EMA-update ``log Z`` from a rejection-loop acceptance rate."""
        if n_drawn == 0:
            return
        z = max(n_kept / n_drawn, 1.0 / (n_drawn + 1.0))
        log_z = math.log(z)
        if self._domain_mass_seen:
            beta = self._domain_mass_ema
            self._log_domain_mass.mul_(1.0 - beta).add_(beta * log_z)
        else:
            self._log_domain_mass.fill_(log_z)
            self._domain_mass_seen = True

    def _mixture_log_prob_from_canonical(
        self,
        canon,
        modes,
        x,
        conj_logdet,
        context=None,
        apply_domain_mass=True,
    ):
        """log q(x) for x generated as ``g_modes . canon``.

        For a group that tiles the space and ``canon`` inside the fundamental
        domain only branch ``modes`` contributes, so ``log q(x) = log pi_modes
        + log q0(canon) + log|det dcanon/du| + conj_logdet``. The base term is
        recomputed here with :meth:`base_flow.log_prob` on the standardised
        ``canon`` -- i.e. exactly branch ``modes`` of :meth:`_branch_log_probs`
        -- rather than reusing the generator's log density at ``u``: the two
        differ by the ``u -> canon -> standardise`` round-off, which a steep
        base density amplifies past tolerance. ``conj_logdet`` is
        ``L(x) - L(canon)``.

        With ``truncate_base_to_domain`` (the default) a draw whose ``canon``
        left the fundamental domain is not one the domain-restricted base flow
        can make, so it gets ``-inf`` and is discarded downstream; the density
        is renormalised by the tracked in-domain base mass ``log Z``. Without
        truncation such a point is instead scored with the full mixture
        :meth:`log_prob`, *floored* at the single-branch value (near a
        degenerate edge :meth:`_branch_log_probs` can place none of the orbit
        images cleanly in-domain and return a spuriously tiny density).

        Independently of truncation, a point whose branch inverse does not map
        ``x`` back onto ``canon`` (a non-injective action -- one that clamps /
        saturates outside a box, e.g. an angle through ``asin(clamp(...))``) is
        floored at the full-mixture density.
        """
        base_lp = self._base_log_prob(canon, modes, context=context)
        log_q = (
            torch.log(self.weights[modes])
            + base_lp
            + self._canon_log_det(modes)
            + conj_logdet
        )
        out_of_domain = torch.zeros(
            x.shape[0], dtype=torch.bool, device=x.device
        )
        if self.in_fundamental_domain is not None or self.uses_prime_space_action:
            out_of_domain = ~self._in_domain(canon)
        pre_rt, _ = self._apply_group_action(x, modes, inverse=True)
        no_roundtrip = ~torch.isclose(
            pre_rt, canon, atol=1e-5, rtol=1e-5
        ).all(dim=-1)
        rt_only = no_roundtrip & ~out_of_domain
        bad = out_of_domain | no_roundtrip
        if bad.numel():
            self._last_leakage_fraction = float(bad.float().mean())
            self._last_leakage_fraction_domain = float(
                out_of_domain.float().mean()
            )
            # Disjoint split: attribute an in-domain point to the round-trip
            # cause only where the domain cause does not already fire.
            self._last_leakage_fraction_roundtrip = float(
                rt_only.float().mean()
            )

        if self._truncate:
            # Feed the domain-leak rate into the EMA estimate of log Z (both
            # the sample_and_log_prob reject loop -- which passes
            # ``apply_domain_mass=False`` and does its own update -- and the
            # backward_pass/inverse path land here).
            if apply_domain_mass and out_of_domain.numel():
                self._update_domain_mass(
                    int((~out_of_domain).sum()), out_of_domain.numel()
                )
            log_q = log_q.clone()
            # Discard non-canonical proposals: an out-of-domain ``canon`` is
            # not a draw the domain-restricted base flow can make, so it gets
            # -inf and callers (backward_pass' finite-log_prob filter, the
            # reject loop) drop it.
            log_q[out_of_domain] = -float("inf")
            if bool(rt_only.any()):
                log_q[rt_only] = torch.maximum(
                    self._raw_log_prob(x[rt_only], context=context),
                    log_q[rt_only],
                )
            if apply_domain_mass:
                log_q = log_q - self._log_domain_mass
            return log_q

        if bool(bad.any()):
            log_q = log_q.clone()
            log_q[bad] = torch.maximum(
                self._raw_log_prob(x[bad], context=context), log_q[bad]
            )
        return log_q

    def sample_and_log_prob(self, num_samples, context=None):
        if self._truncate:
            return self._sample_and_log_prob_truncated(num_samples, context)
        u = self.base_flow.sample(num_samples, context=context)
        modes = Categorical(probs=self.weights).sample((num_samples,))
        t = self._fold_reflect(self._destandardise(u, modes))
        canon, _ = self._from_base(t)
        x, fwd_logdet = self._apply_group_action(
            canon, modes, inverse=False
        )
        log_q = self._mixture_log_prob_from_canonical(
            canon, modes, x, -fwd_logdet, context
        )
        return x, log_q

    def _sample_and_log_prob_truncated(self, num_samples, context=None):
        """``sample_and_log_prob`` rejecting ``canon`` outside the domain.

        Draws base samples in rounds, keeps only those whose canonical
        representative lies in the fundamental domain, and scores them with the
        (now exact) single-branch density minus the EMA-tracked ``log Z``. The
        acceptance rate of the loop feeds the ``log Z`` estimate.
        """
        xs, lqs = [], []
        n_have = n_kept_tot = n_drawn_tot = 0
        max_rounds = 32
        for _ in range(max_rounds):
            need = num_samples - n_have
            if need <= 0:
                break
            # Running acceptance rate (falls back to the EMA estimate, then
            # to 1) sizes the next draw.
            if n_drawn_tot:
                z_guess = max(n_kept_tot / n_drawn_tot, 1e-3)
            else:
                z_guess = float(
                    torch.exp(self._log_domain_mass).clamp(min=1e-3)
                )
            n_draw = int(math.ceil(need / z_guess * 1.3)) + 32
            u = self.base_flow.sample(n_draw, context=context)
            modes = Categorical(probs=self.weights).sample((n_draw,))
            t = self._fold_reflect(self._destandardise(u, modes))
            canon, _ = self._from_base(t)
            in_dom = self._in_domain(canon)
            n_drawn_tot += n_draw
            n_kept_tot += int(in_dom.sum())
            if not bool(in_dom.any()):
                continue
            canon_k, modes_k = canon[in_dom], modes[in_dom]
            x, fwd_logdet = self._apply_group_action(
                canon_k, modes_k, inverse=False
            )
            lq = self._mixture_log_prob_from_canonical(
                canon_k,
                modes_k,
                x,
                -fwd_logdet,
                context,
                apply_domain_mass=False,
            )
            xs.append(x)
            lqs.append(lq)
            n_have += x.shape[0]
        self._update_domain_mass(n_kept_tot, n_drawn_tot)
        if not xs:
            raise RuntimeError(
                "Group-mixture truncated sampling drew no in-domain points: "
                "the base flow has negligible mass in the fundamental domain."
            )
        x = torch.cat(xs)[:num_samples]
        log_q = torch.cat(lqs)[:num_samples] - self._log_domain_mass
        if x.shape[0] < num_samples:
            logger.warning(
                "Group-mixture truncated sampling returned %d of %d requested "
                "samples after %d rounds (domain mass ~%.3g).",
                x.shape[0],
                num_samples,
                max_rounds,
                float(torch.exp(self._log_domain_mass)),
            )
        return x, log_q

    # -- Remaining BaseFlow abstract methods -----------------------------

    def forward(self, x, context=None):
        """Map ``x`` to the latent space via its canonical representative.

        Used for the latent truncation radius. The geometric branch
        assignment costs no base-flow evaluation.
        """
        assigned, pre, _ = self._assign_branch(x)
        canon = pre[assigned, torch.arange(x.shape[0], device=x.device)]
        t, _ = self._to_base(canon)
        return self.base_flow.forward(
            self._standardise(t, assigned), context=context
        )

    def inverse(self, z, context=None):
        # ``log_j`` is set so that
        # ``latent_log_prob(z) - log_j == self.log_prob(x)``, the quantity
        # ``FlowProposal.backward_pass`` relies on. It is not a literal
        # Jacobian: the *physical* group action is assumed measure
        # preserving, and the prime-space conjugation Jacobian is carried in
        # ``fwd_logdet``.
        modes = Categorical(probs=self.weights).sample((z.shape[0],))

        u, _ = self.base_flow.inverse(z, context=context)
        t = self._fold_reflect(self._destandardise(u, modes))
        canon, _ = self._from_base(t)
        x, fwd_logdet = self._apply_group_action(
            canon, modes, inverse=False
        )

        latent_log_prob = self.base_flow.base_distribution_log_prob(
            z, context=context
        )
        log_q = self._mixture_log_prob_from_canonical(
            canon, modes, x, -fwd_logdet, context
        )
        log_j = latent_log_prob - log_q
        return x, log_j

    def sample(self, n, context=None):
        x, _ = self.sample_and_log_prob(n, context=context)
        return x

    def sample_latent_distribution(self, n, context=None):
        return self.base_flow.sample_latent_distribution(n, context=context)

    def base_distribution_log_prob(self, z, context=None):
        return self.base_flow.base_distribution_log_prob(z, context=context)

    def forward_and_log_prob(self, x, context=None):
        z, _ = self.forward(x, context=context)
        return z, self.log_prob(x, context=context)

    def freeze_transform(self):
        self.base_flow.freeze_transform()

    def unfreeze_transform(self):
        self.base_flow.unfreeze_transform()

    def finalise(self):
        self.base_flow.finalise()

    def end_iteration(self):
        self.base_flow.end_iteration()


class GroupMixtureFlowModel(FlowModel):
    """FlowModel that builds a :class:`DiscreteGroupMixtureFlowWrapper`.

    Strips the group-mixture-specific config keys before delegating to
    :func:`~nessai.flows.utils.configure_model` for the base flow.
    """

    group_action_fn = None
    group_size = None
    param_names = None
    in_fundamental_domain = None
    prime_space_action = None
    prime_space_in_domain = None
    min_canon_std = 1e-2
    truncate_base_to_domain = True
    reflect_parameters = None
    canonical_transform = None
    mode_factor_sizes = None

    def initialise(self):
        """Initialise the model and optimiser via :meth:`get_model`."""
        self.update_mask()
        self.model = self.get_model(self.flow_config)
        logger.debug("Flow model:")
        logger.debug(self.model)
        self.device = torch.device(
            self.training_config.get("device_tag", "cpu")
        )
        self.model.device = self.device
        logger.debug(f"Training device: {self.device}")
        self.inference_device = torch.device(
            self.flow_config.get("inference_device_tag", self.device)
            or self.device
        )
        logger.debug(f"Inference device: {self.inference_device}")

        self._optimiser = self.get_optimiser()
        self.initialised = True

    def get_model(self, config):
        """Build the base flow and wrap it for the group mixture."""
        # Shallow copy so the group-mixture keys can be popped without
        # mutating the caller's config or reaching the base constructor.
        config_clean = config.copy()
        config_clean.pop("model", None)
        group_action_fn = config_clean.pop(
            "group_action_fn", getattr(self, "group_action_fn", None)
        )
        group_size = config_clean.pop(
            "group_size", getattr(self, "group_size", None)
        )
        param_names = config_clean.pop(
            "param_names", getattr(self, "param_names", None)
        )
        in_fundamental_domain = config_clean.pop(
            "in_fundamental_domain",
            getattr(self, "in_fundamental_domain", None),
        )
        prime_space_action = config_clean.pop(
            "prime_space_action",
            getattr(self, "prime_space_action", None),
        )
        prime_space_in_domain = config_clean.pop(
            "prime_space_in_domain",
            getattr(self, "prime_space_in_domain", None),
        )
        min_canon_std = config_clean.pop(
            "min_canon_std", getattr(self, "min_canon_std", 1e-2)
        )
        truncate_base_to_domain = config_clean.pop(
            "truncate_base_to_domain",
            getattr(self, "truncate_base_to_domain", True),
        )
        reflect_parameters = config_clean.pop(
            "reflect_parameters", getattr(self, "reflect_parameters", None)
        )
        canonical_transform = config_clean.pop(
            "canonical_transform",
            getattr(self, "canonical_transform", None),
        )
        mode_factor_sizes = config_clean.pop(
            "mode_factor_sizes", getattr(self, "mode_factor_sizes", None)
        )
        if not isinstance(mode_factor_sizes, (list, tuple)):
            mode_factor_sizes = None

        if group_action_fn is None or group_size is None:
            raise ValueError(
                "GroupMixtureFlowModel requires `group_action_fn` and "
                "`group_size`."
            )

        base_flow = configure_model(config_clean)
        num_features = config_clean.get("n_inputs")

        return DiscreteGroupMixtureFlowWrapper(
            base_flow=base_flow,
            num_features=num_features,
            group_action_fn=group_action_fn,
            group_size=group_size,
            param_names=param_names,
            in_fundamental_domain=in_fundamental_domain,
            prime_space_action=prime_space_action,
            prime_space_in_domain=prime_space_in_domain,
            min_canon_std=min_canon_std,
            truncate_base_to_domain=truncate_base_to_domain,
            reflect_parameters=reflect_parameters,
            canonical_transform=canonical_transform,
            mode_factor_sizes=mode_factor_sizes,
        )


def make_group_mixture_flow(
    group_action_fn,
    group_size,
    param_names,
    in_fundamental_domain=None,
    prime_space_action=None,
    prime_space_in_domain=None,
    min_canon_std=1e-2,
    truncate_base_to_domain=True,
    reflect_parameters=None,
    canonical_transform=None,
    mode_factor_sizes=None,
):
    """Factory constructing a ``GroupMixtureFlowModel`` bound to a specific group.

    Parameters
    ----------
    group_action_fn : callable
        ``(point_dict, modes, inverse=False) -> point_dict`` applying the
        group action, in the *physical* parameter space. A non-measure-
        preserving action may instead return ``(point_dict, log_det)`` with
        ``log_det`` the log-determinant of the physical map it applied.
    group_size : int
        Number of discrete group elements.
    param_names : list of str
        Ordered physical parameter names.
    in_fundamental_domain : callable, optional
        ``(point_dict) -> bool tensor`` marking whether each point is the
        canonical orbit representative, defined in the physical parameter
        space. If omitted the mixture weights are not identifiable (the
        base flow can absorb the whole distribution).
    prime_space_action : callable, optional
        ``(point_dict, modes, inverse=False) -> point_dict`` or
        ``-> (point_dict, log_det)`` applying the action directly in the
        flow's *prime* coordinates. When given, the coordinate bridge is
        bypassed and ``group_action_fn`` / ``in_fundamental_domain`` are
        ignored. ``log_det`` is the log-Jacobian of the prime-space map
        (default 0 for a measure-preserving action such as a rotation);
        use this for augmented / dimension-changing reparameterisations
        (``Angle``, ``AnglePair``) where the automatic bridge cannot round
        trip.
    prime_space_in_domain : callable, optional
        Fundamental-domain predicate in prime coordinates; required with
        ``prime_space_action``.
    min_canon_std : float, optional
        Floor on the per-element canonical standardisation std (default
        ``1e-2``, tuned for a ~unit-scaled prime frame). Lower it when a
        parameter is z-scored far tighter than the prior resolves it (e.g. a
        GW ``geocent_time``), otherwise the base flow is forced to model a
        near-delta in that coordinate and ``sample_and_log_prob`` is
        over-dispersed there; a one-off warning fires when the floor binds.
    truncate_base_to_domain : bool, optional
        If ``True``,
        :meth:`~DiscreteGroupMixtureFlowWrapper.sample_and_log_prob` rejects
        any generative draw whose canonical representative falls outside the
        fundamental domain rather than scoring it with the leaky
        single-branch shortcut. The base density becomes ``q0`` restricted to
        the domain and renormalised by its mass there (tracked as an EMA of the
        rejection acceptance rate), which makes the single-branch density exact
        and removes the importance-weight bias from base-flow mass leaking into
        neighbouring tiles. Default ``True``; a no-op (with an info log) when
        no fundamental-domain predicate is available.
    reflect_parameters : list of str, optional
        Prime-space coordinate names (a subset of ``param_names``) whose
        fundamental domain is bounded by a hard wall at 0 -- e.g. the
        ``x, y, z >= 0`` octant faces of a detector-frame sky decomposition.
        The base flow then models the sign-symmetric extension
        ``q0_sym(u) = sum_s q0(s . u)`` over the ``2**m`` sign patterns of
        these dims (so it never has to represent the wall cliff), the
        per-element standardisation of these dims is pinned to mean 0 / RMS
        scale, and a generative draw is folded back with ``abs``. Only
        supported on the ``prime_space_action`` path. Default: no reflection.
    canonical_transform : object, optional
        Fixed analytic bijection between the canonical prime coordinates and
        the coordinates the base flow models (dimension-preserving), exposing
        ``forward(canon) -> (t, log|det dt/dcanon|)``,
        ``inverse(t) -> (canon, log|det dcanon/dt|)`` (batched torch) and an
        optional ``bind(param_names)``. Applied inside the fundamental domain,
        so the group action and domain predicate are untouched; use it to turn
        a hard canonical geometry (a uniform sky octant with a sharp prior
        edge) into a near-Gaussian frame before standardisation. Only
        supported on the ``prime_space_action`` path. Default: identity.
    mode_factor_sizes : list of int, optional
        Sizes ``[s_0, ..., s_{F-1}]`` (product ``== group_size``) of the
        commuting cyclic factors the group decomposes into, with the mode
        index a little-endian mixed-radix code
        (``factor_f(g) = (g // prod(s_{<f})) % s_f``). When given,
        :meth:`~DiscreteGroupMixtureFlowWrapper.update_mixture_weights`
        estimates the ``F`` factor marginals independently -- each pooling
        assignment counts over every other factor -- and sets the weights to
        their product, which is much more robust to transient mode collapse
        than the flat per-mode count (a starved joint mode keeps a non-zero
        weight until a whole marginal slice empties). Default: flat per-mode
        estimator.

    Notes
    -----
    ``group_action_fn`` and ``in_fundamental_domain`` are defined in the
    physical parameter space, so the proposal class must use
    :class:`GroupFlowProposalMixin`. For an affine reparameterisation this
    is exact and free; for a non-affine one the mixin installs a
    :class:`ReparamBridge` that round-trips through the reparameterisation
    in numpy each batch and carries the exact conjugation Jacobian.
    """

    class CustomGroupMixtureFlowModel(GroupMixtureFlowModel):
        pass

    CustomGroupMixtureFlowModel.group_action_fn = staticmethod(group_action_fn)
    CustomGroupMixtureFlowModel.group_size = group_size
    CustomGroupMixtureFlowModel.param_names = param_names
    CustomGroupMixtureFlowModel.min_canon_std = min_canon_std
    CustomGroupMixtureFlowModel.truncate_base_to_domain = (
        truncate_base_to_domain
    )
    CustomGroupMixtureFlowModel.reflect_parameters = (
        list(reflect_parameters) if reflect_parameters else None
    )
    CustomGroupMixtureFlowModel.canonical_transform = canonical_transform
    CustomGroupMixtureFlowModel.mode_factor_sizes = (
        [int(s) for s in mode_factor_sizes]
        if mode_factor_sizes is not None
        else None
    )
    if in_fundamental_domain is not None:
        CustomGroupMixtureFlowModel.in_fundamental_domain = staticmethod(
            in_fundamental_domain
        )
    if prime_space_action is not None:
        CustomGroupMixtureFlowModel.prime_space_action = staticmethod(
            prime_space_action
        )
    if prime_space_in_domain is not None:
        CustomGroupMixtureFlowModel.prime_space_in_domain = staticmethod(
            prime_space_in_domain
        )
    return CustomGroupMixtureFlowModel


# ---------------------------------------------------------------------------
# Clustered group mixture: K independent group-mixture base flows, one per
# data cluster, each with its own per-branch canonical standardisation and
# latent ball. Reclustered each training round (GMM, largest k in
# [1, n_clusters_max] with well-separated components), so k tracks the folded
# posterior as it goes unimodal -> bimodal over the run; k == 1 reproduces the
# single-flow behaviour exactly.
# ---------------------------------------------------------------------------
def _farthest_point_seeds(ts, existing, m):
    """``m`` rows of ``ts`` maximising the minimum distance to ``existing`` (and
    to already-picked seeds) -- a k-means++-style spread used to place the
    centres of *newly added* clusters when ``k`` grows."""
    m = int(m)
    if m <= 0:
        return np.empty((0, ts.shape[1]))
    picks = []
    base = list(np.atleast_2d(np.asarray(existing, dtype=float)))
    for _ in range(m):
        ref = np.asarray(base + picks)
        d = np.linalg.norm(
            ts[:, None, :] - ref[None, :, :], axis=2
        ).min(axis=1)
        picks.append(ts[int(np.argmax(d))])
    return np.asarray(picks)


class ClusteredGroupMixtureFlowWrapper(BaseFlow):
    """Mixture over ``K`` :class:`DiscreteGroupMixtureFlowWrapper` experts.

    ``log q(x) = logsumexp_j [log w_j + log q_j(x)]`` with ``w_j`` the cluster
    fraction and ``q_j`` the ``j``-th expert (a full group mixture with its
    own canonical standardisation).  Only ``_n_active`` experts contribute;
    the rest carry weight 0 and are not trained.  With ``_n_active == 1`` every
    method is exactly ``experts[0]`` (the current single-flow behaviour).

    Routing (which expert scores/generates a point, used for the per-cluster
    training loss and the latent-radius geometry) is nearest-centroid in the
    standardised base-flow frame -- the frame ``q0`` sees, after the branch
    fold and the canonical transform.  The sklearn clusterer runs only in
    :meth:`update_mixture_weights` to pick ``k`` and seed the centroids;
    everything downstream is the centroid buffers, so the wrapper pickles and
    resumes with the rest of the flow ``state_dict``.
    """

    def __init__(
        self,
        experts,
        num_features,
        *,
        cluster_method="gmm",
        max_cluster_overlap=0.05,
        min_cluster_size=200,
        weight_ema=0.5,
        k_shrink_patience=3,
        k_grow_patience=2,
        centroid_ema=None,
        bg_expert=None,
        bg_weight=0.0,
    ):
        super().__init__()
        self.experts = torch.nn.ModuleList(experts)
        self.n_experts = len(experts)
        # Optional always-on "background" expert: a full group-mixture flow
        # trained every round on *all* the data (not routed), blended into the
        # generative / density path at a fixed weight whenever ``k >= 2``.  It
        # floors the mixture density everywhere the live points are, so no
        # importance weight can blow up where the per-cluster experts leave
        # off.  ``bg_weight == 0`` (the default) leaves ``_bg_expert is None``
        # and every path byte-identical to the plain clustered mixture.
        self._bg_expert = bg_expert if float(bg_weight) > 0.0 else None
        self.register_buffer(
            "_bg_weight", torch.tensor(float(bg_weight))
        )
        self.register_buffer("_bg_seen", torch.zeros((), dtype=torch.bool))
        self.num_features = int(num_features)
        self.group_size = experts[0].group_size
        self.cluster_method = cluster_method
        # k hysteresis: a *rise* only takes after this many consecutive rounds
        # want more clusters (so a one-round spurious split does not spawn --
        # then, ``k_shrink_patience`` rounds later, discard -- an expert), and a
        # *drop* only after ``k_shrink_patience`` want fewer.
        self.k_shrink_patience = int(k_shrink_patience)
        self.k_grow_patience = max(int(k_grow_patience), 1)
        # EMA weight for the per-cluster routing centroids across rounds (in raw
        # base-flow-frame units), when ``k`` is unchanged.  Smooths the routing
        # boundary so points near it stop hopping between experts round to
        # round.  Defaults to ``weight_ema``.
        self.centroid_ema = (
            float(weight_ema) if centroid_ema is None else float(centroid_ema)
        )
        # Accept a k-way split only if it is *well separated*: at most this
        # fraction of points sit in the fuzzy zone between clusters (GMM
        # responsibility < 0.8).  Splitting a single blob makes ~half of it
        # ambiguous, so this naturally rejects over-splitting; 0 <-> one flow.
        self.max_cluster_overlap = float(max_cluster_overlap)
        self.min_cluster_size = int(min_cluster_size)
        self.weight_ema = float(weight_ema)

        w0 = torch.zeros(self.n_experts)
        w0[0] = 1.0
        self.register_buffer("cluster_weights", w0)
        self.register_buffer("_n_active", torch.tensor(1, dtype=torch.long))
        self.register_buffer(
            "_centroids", torch.zeros(self.n_experts, self.num_features)
        )
        self.register_buffer("_base_mu", torch.zeros(self.num_features))
        self.register_buffer("_base_sd", torch.ones(self.num_features))
        self.register_buffer(
            "_k_shrink_streak", torch.zeros((), dtype=torch.long)
        )
        self.register_buffer(
            "_k_grow_streak", torch.zeros((), dtype=torch.long)
        )
        self.register_buffer(
            "_clustering_seen", torch.zeros((), dtype=torch.bool)
        )
        # True between a k-increase and the first training pass that follows
        # it: while set, every active expert is kept a byte-identical copy of
        # expert 0 (standardised on the *full* data), so the mixture density
        # is exactly the pre-split single-flow density and the flip costs no
        # population acceptance.  The per-cluster loss specialises the experts
        # during that training; :meth:`finalise` then clears the flag.
        self.register_buffer(
            "_pending_split_train", torch.zeros((), dtype=torch.bool)
        )
        self._cluster_cache = None  # (data_ptr, n, labels tensor)

    # -- pass-throughs the proposal / diagnostics expect ----------------
    @property
    def uses_prime_space_action(self):
        return self.experts[0].uses_prime_space_action

    @property
    def _min_canon_std(self):
        return self.experts[0]._min_canon_std

    @property
    def mode_factor_sizes(self):
        return self.experts[0].mode_factor_sizes

    @property
    def _weight_empty_patience(self):
        return self.experts[0]._weight_empty_patience

    @property
    def _factor_empty_rounds(self):
        return self.experts[0]._factor_empty_rounds

    @property
    def _last_factor_marginals(self):
        return self.experts[0]._last_factor_marginals

    @property
    def _last_factor_counts(self):
        return self.experts[0]._last_factor_counts

    @property
    def weights(self):
        # Group-element mixture weights of the routed/first expert. Read by
        # ``GroupFlowProposalMixin._training_data_as_prime_tensor`` for the
        # tensor dtype/device and by diagnostics; the clustered weights are
        # ``cluster_weights``.
        return self.experts[0].weights

    @property
    def param_names(self):
        return self.experts[0].param_names

    def set_param_names(self, names):
        for e in self._all_experts():
            e.set_param_names(names)

    def set_coordinate_bridge(self, bridge):
        for e in self._all_experts():
            e.set_coordinate_bridge(bridge)

    def set_affine_maps(self, scale, shift):
        for e in self._all_experts():
            e.set_affine_maps(scale, shift)

    def _assign_branch(self, x):
        return self.experts[0]._assign_branch(x)

    def _to_base(self, canon):
        return self.experts[0]._to_base(canon)

    # -- routing --------------------------------------------------------
    def _fold_to_base(self, x):
        e = self.experts[0]
        with torch.no_grad():
            assigned, pre, _ = e._assign_branch(x)
            canon = pre[assigned, torch.arange(x.shape[0], device=x.device)]
            t, _ = e._to_base(canon)
        return t

    @torch.no_grad()
    def route_prime_array(self, samples):
        """Nearest-centroid expert assignment for an unstructured prime-space
        array -- the training data :meth:`FlowModel.train` receives."""
        x = torch.as_tensor(
            samples, dtype=self.weights.dtype, device=self.weights.device
        )
        return self._route(x).cpu().numpy()

    @torch.no_grad()
    def _route(self, x):
        act = int(self._n_active.item())
        if act <= 1 or not bool(self._clustering_seen):
            return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        ts = (self._fold_to_base(x) - self._base_mu) / self._base_sd
        return torch.cdist(ts, self._centroids[:act]).argmin(dim=1)

    # -- densities -----------------------------------------------------
    def _active(self):
        """Experts contributing to the *generative / density* path.

        Collapses to 1 while a k-increase is pending its first training pass:
        ``populate()`` then only ever sees the (safe) pre-split single-flow
        density -- no acceptance cliff -- while the per-cluster loss still
        specialises all ``k`` experts during that training (see
        :meth:`_n_active_experts`).
        """
        if bool(self._pending_split_train.item()):
            return 1
        return int(self._n_active.item())

    def _n_active_experts(self):
        """The real ``k`` -- experts that train per-cluster, even while a
        split is pending."""
        return int(self._n_active.item())

    def _bg_on(self):
        """True when the background expert should contribute to the mixture.

        Only for ``k >= 2``: at ``k == 1`` ``experts[0]`` is already the
        full-data flow, so the background component would be redundant and the
        (byte-identical) single-flow fast path is kept.
        """
        return (
            self._bg_expert is not None
            and float(self._bg_weight) > 0.0
            and self._active() >= 2
        )

    def _all_experts(self):
        """Cluster experts plus the background expert (if any) -- for
        lifecycle calls (param names, bridge, freeze, finalise, ...)."""
        if self._bg_expert is None:
            return list(self.experts)
        return list(self.experts) + [self._bg_expert]

    def _blend_bg(self, fg_log_prob, x, context=None):
        """logaddexp the background density into a foreground log-prob."""
        if not self._bg_on():
            return fg_log_prob
        w = float(self._bg_weight)
        bg = self._bg_expert.log_prob(x, context=context)
        return torch.logaddexp(
            fg_log_prob + math.log1p(-w), bg + math.log(w)
        )

    def log_prob(self, x, context=None):
        act = self._active()
        if act == 1:
            return self.experts[0].log_prob(x, context=context)
        lw = torch.log(self.cluster_weights[:act].clamp_min(1e-38))
        lps = torch.stack(
            [
                self.experts[j].log_prob(x, context=context) + lw[j]
                for j in range(act)
            ],
            dim=0,
        )
        return self._blend_bg(torch.logsumexp(lps, dim=0), x, context=context)

    def base_distribution_log_prob(self, z, context=None):
        return self.experts[0].base_distribution_log_prob(z, context=context)

    def sample_latent_distribution(self, n, context=None):
        return self.experts[0].sample_latent_distribution(n, context=context)

    # -- generative --------------------------------------------------
    def _draw_assignments(self, n, device):
        act = self._active()
        w = self.cluster_weights[:act]
        return torch.multinomial(w / w.sum(), int(n), replacement=True).to(
            device
        )

    def sample_and_log_prob(self, num_samples, context=None):
        act = self._active()
        if act == 1:
            return self.experts[0].sample_and_log_prob(
                num_samples, context=context
            )
        dev = self.cluster_weights.device
        n_bg = 0
        if self._bg_on():
            n_bg = int(
                torch.binomial(
                    torch.tensor(float(num_samples)),
                    torch.tensor(float(self._bg_weight)),
                ).item()
            )
        assign = self._draw_assignments(num_samples - n_bg, dev)
        parts = []
        for j in range(act):
            nj = int((assign == j).sum())
            if nj:
                xj, _ = self.experts[j].sample_and_log_prob(
                    nj, context=context
                )
                parts.append(xj)
        if n_bg:
            xb, _ = self._bg_expert.sample_and_log_prob(n_bg, context=context)
            parts.append(xb)
        x = torch.cat(parts, dim=0)
        x = x[torch.randperm(x.shape[0], device=x.device)]
        return x, self.log_prob(x, context=context)

    def sample(self, n, context=None):
        return self.sample_and_log_prob(n, context=context)[0]

    def forward(self, x, context=None):
        act = self._active()
        if act == 1:
            return self.experts[0].forward(x, context=context)
        r = self._route(x)
        z = x.new_zeros(x.shape[0], self.num_features)
        log_j = x.new_zeros(x.shape[0])
        for j in range(act):
            m = r == j
            if bool(m.any()):
                zj, jj = self.experts[j].forward(x[m], context=context)
                z[m] = zj
                log_j[m] = jj
        return z, log_j

    def inverse(self, z, context=None):
        act = self._active()
        if act == 1:
            return self.experts[0].inverse(z, context=context)
        n = z.shape[0]
        x = z.new_zeros(n, self.num_features)
        # Each expert is a domain-truncated flow: ~20-30 % of its base N(0, I)
        # mass maps to a canonical representative *outside* the fundamental
        # domain, and ``DiscreteGroupMixtureFlowWrapper.inverse`` flags those
        # draws with a non-finite ``log_j`` so ``backward_pass`` discards them
        # (exactly what its rejection-based ``sample_and_log_prob`` does).
        # Recomputing ``log_q`` below from :meth:`log_prob` alone would
        # resurrect them with a small *finite* raw density -> a fat
        # importance-weight tail that collapses ``populate()``.  Carry the
        # per-expert (and background) domain rejection through.
        gen_out_of_domain = torch.zeros(n, dtype=torch.bool, device=z.device)
        if self._bg_on():
            use_bg = torch.rand(n, device=z.device) < float(self._bg_weight)
        else:
            use_bg = torch.zeros(n, dtype=torch.bool, device=z.device)
        fg = (~use_bg).nonzero(as_tuple=True)[0]
        if fg.numel():
            assign = self._draw_assignments(fg.numel(), z.device)
            for j in range(act):
                sub = fg[assign == j]
                if sub.numel():
                    xj, ljj = self.experts[j].inverse(z[sub], context=context)
                    x[sub] = xj
                    gen_out_of_domain[sub] = ~torch.isfinite(ljj)
        if bool(use_bg.any()):
            xb, ljb = self._bg_expert.inverse(z[use_bg], context=context)
            x[use_bg] = xb
            gen_out_of_domain[use_bg] = ~torch.isfinite(ljb)
        # non-literal log_j: base_distribution_log_prob(z) - log_j == log q(x)
        log_q = self.log_prob(x, context=context)
        if bool(gen_out_of_domain.any()):
            log_q = log_q.clone()
            log_q[gen_out_of_domain] = -float("inf")
        log_j = (
            self.base_distribution_log_prob(z, context=context) - log_q
        )
        return x, log_j

    def forward_and_log_prob(self, x, context=None):
        z, _ = self.forward(x, context=context)
        return z, self.log_prob(x, context=context)

    def freeze_transform(self):
        for e in self._all_experts():
            e.freeze_transform()

    def unfreeze_transform(self):
        for e in self._all_experts():
            e.unfreeze_transform()

    def finalise(self):
        for e in self._all_experts():
            e.finalise()
        # A full training pass at the new k has just completed: the
        # per-cluster loss has specialised the experts, so release the
        # standardisation freeze and let subsequent rounds track each
        # cluster independently.
        if bool(self._pending_split_train.item()) and (
            self._n_active_experts() >= 2
        ):
            self._pending_split_train.fill_(False)
            logger.info(
                "Clustered group mixture: split trained, all %d experts now "
                "contribute to the proposal",
                self._n_active_experts(),
            )

    def end_iteration(self):
        for e in self._all_experts():
            e.end_iteration()

    # -- per-cluster training loss ----------------------------------
    def loss_function(self, x, conditional=None):
        """Per-cluster negative log-likelihood.

        Each point is scored only by its routed expert; the weighted sum is
        the mixture NLL with a hard assignment (the piecewise-flow objective
        of arXiv:2305.02930).  ``FlowModel._train`` picks this up via
        ``hasattr(model, "loss_function")``.
        """
        act = self._n_active_experts()
        if act == 1:
            return -self.experts[0].log_prob(x).mean()
        r = self._route(x)
        total = x.new_zeros(())
        for j in range(act):
            m = r == j
            if bool(m.any()):
                total = total + self.cluster_weights[j] * (
                    -self.experts[j].log_prob(x[m]).mean()
                )
        return total

    # -- clustering update (called from GroupFlowProposalMixin.check_state)
    def _cluster(self, x):
        """Recluster the folded training data; return the per-row labels.

        The number of clusters evolves over the run: an initially unimodal
        folded posterior gives ``k = 1`` (the plain single flow), and once it
        becomes decisively multi-modal ``k`` rises and the new expert --
        warm-started from the first -- begins specialising.  ``k`` is
        regularised so it (and the cluster<->expert mapping) does not jitter
        round to round:

          * a *rise* in ``k`` takes only after :attr:`k_grow_patience`
            consecutive rounds want it, a *drop* only after
            :attr:`k_shrink_patience` (hysteresis both ways);
          * the GMM is warm-started from the previous round's centroids
            (:meth:`_fit_gmm` ``means_init``) so point membership is stable;
          * clusters are matched to the previous round's centroids by optimal
            assignment (:meth:`_match_clusters`), so a physical sheet keeps its
            expert slot instead of swapping;
          * the routing centroids are carried with an EMA
            (:attr:`centroid_ema`) while ``k`` is unchanged.
        """
        # Fingerprint the data so the two back-to-back calls from
        # ``check_state`` (update_mixture_weights then
        # update_base_standardisation) reuse one clustering, without keying on
        # ``data_ptr`` alone (memory addresses get recycled between rounds).
        key = (
            tuple(x.shape),
            float(x.sum().item()),
            float(x.reshape(-1)[:: max(x.numel() // 32, 1)].sum().item()),
        )
        if self._cluster_cache is not None and self._cluster_cache[0] == key:
            return self._cluster_cache[1]

        t = self._fold_to_base(x).detach().cpu().numpy()
        n = t.shape[0]
        mu = t.mean(axis=0)
        sd = np.maximum(t.std(axis=0), 1e-9)
        ts = (t - mu) / sd

        k_max = min(self.n_experts, n // max(self.min_cluster_size, 1))
        k_want = self._choose_k(ts, k_max) if k_max >= 2 else 1
        k_cur = int(self._n_active.item())

        if not bool(self._clustering_seen):
            k = k_want
        elif k_want > k_cur:
            self._k_grow_streak += 1
            self._k_shrink_streak.zero_()
            if int(self._k_grow_streak.item()) >= self.k_grow_patience:
                k = k_want
                self._k_grow_streak.zero_()
            else:
                k = k_cur
        elif k_want < k_cur:
            self._k_shrink_streak += 1
            self._k_grow_streak.zero_()
            if int(self._k_shrink_streak.item()) >= self.k_shrink_patience:
                k = k_want
                self._k_shrink_streak.zero_()
            else:
                k = k_cur
        else:
            k = k_cur
            self._k_shrink_streak.zero_()
            self._k_grow_streak.zero_()

        # Previous-round routing centroids in raw base-flow-frame units
        # (``_base_mu`` / ``_base_sd`` still hold last round's normalisation --
        # they are overwritten below).  ``None`` on the first clustering.
        prev_raw = None
        if bool(self._clustering_seen) and k_cur >= 1:
            prev_raw = (
                self._centroids[:k_cur].detach().cpu().numpy()
                * self._base_sd.detach().cpu().numpy()
                + self._base_mu.detach().cpu().numpy()
            )

        # Warm-start the GMM from the previous centroids so cluster membership
        # is stable round to round -- a cold ``n_init=3`` fit hops between the
        # ~equivalent local optima as the training data drifts by a few points.
        means_init = None
        if prev_raw is not None and k >= 2:
            seed = (prev_raw - mu) / sd
            if k > k_cur:
                seed = np.vstack(
                    [seed, _farthest_point_seeds(ts, seed, k - k_cur)]
                )
            means_init = seed[:k]
        labels, centroids = self._fit_gmm(ts, k, means_init=means_init)

        # Keep each cluster with the expert slot whose *previous-round* centroid
        # it is nearest to (optimal assignment), so a physical sheet does not
        # swap experts between rounds -- a swap attaches a trained flow and its
        # per-cluster standardisation to the wrong sub-population and throws the
        # mixture density off by hundreds of nats on the swapped fraction.
        # Tightest-first only on the first clustering (no history yet).
        if k >= 2:
            spreads = np.array([
                ts[labels == c].std() if np.any(labels == c) else np.inf
                for c in range(k)
            ])
            if prev_raw is not None:
                order = self._match_clusters(
                    centroids, prev_raw, mu, sd, spreads
                )
            else:
                order = np.argsort(spreads)
            remap = {old: new for new, old in enumerate(order)}
            labels = np.array([remap[c] for c in labels])
            centroids = centroids[order]

        # Smooth the routing centroids across rounds when k is unchanged, so
        # points near a cluster boundary stop flip-flopping between experts.
        if (
            prev_raw is not None
            and k == k_cur
            and k >= 1
            and 0.0 <= self.centroid_ema < 1.0
        ):
            b = self.centroid_ema
            cent_raw = centroids * sd + mu
            cent_raw = (1.0 - b) * prev_raw + b * cent_raw
            centroids = (cent_raw - mu) / sd

        with torch.no_grad():
            # warm-start any newly activated expert's *flow weights* from
            # expert 0 (a sane starting point) but re-bootstrap its
            # per-cluster canonical standardisation, and mark the split
            # pending: until the first training pass at the new k, the
            # generative / density path (populate) runs on expert 0 alone
            # (``_active() == 1``), so there is no acceptance cliff, while the
            # per-cluster loss specialises every expert in its own frame
            # during that training.  ``finalise`` then clears the flag.
            if k > k_cur and bool(self._clustering_seen):
                src = self.experts[0].state_dict()
                for j in range(max(k_cur, 1), k):
                    self.experts[j].load_state_dict(src)
                    self.experts[j]._canon_seen.zero_()
                    self.experts[j]._domain_mass_seen = False
                self._pending_split_train.fill_(True)
            if k <= 1:
                self._pending_split_train.fill_(False)

            dev = self.cluster_weights.device
            self._base_mu.copy_(torch.as_tensor(mu, dtype=self._base_mu.dtype))
            self._base_sd.copy_(torch.as_tensor(sd, dtype=self._base_sd.dtype))
            self._centroids.zero_()
            self._centroids[:k].copy_(
                torch.as_tensor(centroids, dtype=self._centroids.dtype)
            )
            counts = np.bincount(labels, minlength=self.n_experts).astype(float)
            w = torch.zeros(
                self.n_experts, device=dev, dtype=self.cluster_weights.dtype
            )
            w[:k] = torch.as_tensor(
                counts[:k] / counts[:k].sum(),
                dtype=self.cluster_weights.dtype,
            )
            if not bool(self._clustering_seen) or k != k_cur:
                self.cluster_weights.copy_(w)
            else:
                beta = self.weight_ema
                self.cluster_weights.mul_(1 - beta).add_(beta * w)
                self.cluster_weights.div_(self.cluster_weights.sum())
            self._n_active.fill_(k)
            self._clustering_seen.fill_(True)

        lab_t = torch.as_tensor(labels, device=x.device, dtype=torch.long)
        self._cluster_cache = (key, lab_t)
        if k != k_cur:
            logger.info(
                "Clustered group mixture: k %d -> %d (want %d)",
                k_cur, k, k_want,
            )
        elif k_want != k_cur:
            logger.info(
                "Clustered group mixture: k held at %d (want %d; "
                "grow streak %d/%d, shrink streak %d/%d)",
                k, k_want,
                int(self._k_grow_streak.item()), self.k_grow_patience,
                int(self._k_shrink_streak.item()), self.k_shrink_patience,
            )
        logger.info(
            "Clustered group mixture: k=%d, weights=%s, sizes=%s",
            k,
            np.round(self.cluster_weights[:k].cpu().numpy(), 3).tolist(),
            np.bincount(labels, minlength=k).tolist(),
        )
        return lab_t

    def _match_clusters(self, centroids_std, prev_raw, mu, sd, spreads):
        """``order`` (length ``k``): the new GMM component to place in each
        expert slot ``0..k-1``.

        Optimal (minimum total distance) assignment of this round's components
        to the previous round's centroids, in the current standardised frame.
        When ``k`` grew, the components with no previous match take the
        remaining slots tightest-first (``spreads`` = per-component std); when
        ``k`` shrank, only the closest surviving slots are filled.
        """
        from scipy.optimize import linear_sum_assignment

        k = centroids_std.shape[0]
        prev_now = (np.asarray(prev_raw, dtype=float) - mu) / sd  # (k_cur, d)
        cost = np.linalg.norm(
            centroids_std[:, None, :] - prev_now[None, :, :], axis=2
        )  # (k, k_cur)
        comp_ind, slot_ind = linear_sum_assignment(cost)
        order = [None] * k
        matched = set()
        for c, s in zip(comp_ind, slot_ind):
            if s < k:
                order[s] = int(c)
                matched.add(int(c))
        leftovers = sorted(
            (c for c in range(k) if c not in matched),
            key=lambda c: spreads[c],
        )
        empty = [s for s in range(k) if order[s] is None]
        for s, c in zip(empty, leftovers):
            order[s] = c
        return np.asarray(order, dtype=int)

    def _choose_k(self, ts, k_max):
        """Largest ``k`` in ``[1, k_max]`` whose GMM clusters are all above
        ``min_cluster_size`` and *well separated* -- fewer than
        ``max_cluster_overlap`` of points ambiguous (max responsibility < 0.8).
        Over-splitting a single blob makes ~half of it ambiguous, so it is
        rejected; returns 1 when no split is clean."""
        from sklearn.mixture import GaussianMixture

        k = 1
        for kk in range(2, k_max + 1):
            gm = GaussianMixture(
                n_components=kk, covariance_type="full", n_init=3,
                random_state=0, reg_covar=1e-4,
            ).fit(ts)
            resp = gm.predict_proba(ts)
            lab = resp.argmax(axis=1)
            if np.bincount(lab, minlength=kk).min() < self.min_cluster_size:
                continue
            if float(np.mean(resp.max(axis=1) < 0.8)) > self.max_cluster_overlap:
                continue
            k = kk
        return k

    def _fit_gmm(self, ts, k, means_init=None):
        """``(labels, centroids)`` for a ``k``-cluster fit (``k == 1`` trivial).

        ``means_init`` (k, d), when finite, warm-starts the GMM with a single
        EM run from those centres instead of a cold ``n_init=3`` search -- this
        is what carries cluster identity across training rounds.
        """
        n = ts.shape[0]
        if k <= 1:
            return np.zeros(n, dtype=int), ts.mean(0, keepdims=True)
        from sklearn.cluster import KMeans
        from sklearn.mixture import GaussianMixture

        warm = (
            means_init is not None
            and np.shape(means_init) == (k, ts.shape[1])
            and np.all(np.isfinite(means_init))
        )
        if warm:
            gm = GaussianMixture(
                n_components=k, covariance_type="full", n_init=1,
                means_init=np.asarray(means_init, dtype=float),
                random_state=0, reg_covar=1e-4,
            ).fit(ts)
        else:
            gm = GaussianMixture(
                n_components=k, covariance_type="full", n_init=3,
                random_state=0, reg_covar=1e-4,
            ).fit(ts)
        if self.cluster_method == "kmeans":
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(ts)
            return km.labels_, km.cluster_centers_
        return gm.predict(ts), gm.means_

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Forward compatibility: a checkpoint written by an older build can
        # predate buffers added since (``_pending_split_train``, the
        # k-evolution buffers).  Fill any missing buffer with its current
        # default so ``strict=True`` resume still works.
        sd = dict(state_dict)
        for name, val in super().state_dict().items():
            sd.setdefault(name, val)
        return super().load_state_dict(sd, strict=strict, assign=assign)

    @torch.no_grad()
    def update_mixture_weights(self, x, context=None, smoothing=1.0):
        labels = self._cluster(x)
        pending = bool(self._pending_split_train.item())
        for j in range(self._n_active_experts()):
            # while a split is pending, expert 0 keeps tracking the *full*
            # data (populate() still runs on it alone, so it must stay the
            # pre-split single flow); the new experts bootstrap on their
            # cluster so the next training can specialise them in the right
            # frame.
            xj = x if (pending and j == 0) else x[labels == j]
            if xj.shape[0]:
                self.experts[j].update_mixture_weights(
                    xj, context=context, smoothing=smoothing
                )
        if self._bg_expert is not None and x.shape[0]:
            # the background expert always tracks the full data
            self._bg_expert.update_mixture_weights(
                x, context=context, smoothing=smoothing
            )

    @torch.no_grad()
    def update_base_standardisation(self, x, context=None):
        labels = self._cluster(x)
        pending = bool(self._pending_split_train.item())
        for j in range(self._n_active_experts()):
            xj = x if (pending and j == 0) else x[labels == j]
            if xj.shape[0]:
                self.experts[j].update_base_standardisation(
                    xj, context=context
                )
        if self._bg_expert is not None and x.shape[0]:
            self._bg_expert.update_base_standardisation(x, context=context)


class ClusteredGroupMixtureFlowModel(GroupMixtureFlowModel):
    """:class:`GroupMixtureFlowModel` building a
    :class:`ClusteredGroupMixtureFlowWrapper` of ``n_clusters_max`` experts."""

    n_clusters_max = 2
    cluster_method = "gmm"
    max_cluster_overlap = 0.05
    min_cluster_size = 200
    k_shrink_patience = 3
    k_grow_patience = 2
    centroid_ema = None
    # Weight of an always-on background expert (trained on all data, blended in
    # at ``k >= 2``).  0 -> no background expert (default; byte-identical to the
    # plain clustered mixture).
    bg_weight = 0.0

    def train(self, samples, weights=None, conditional=None, plot=True,
              **kwargs):
        """Train each active expert independently on its own cluster.

        The joint objective
        (:meth:`ClusteredGroupMixtureFlowWrapper.loss_function`) shares one
        optimiser and one early stop across all experts, so a
        slower-converging expert is cut short the moment the (dominant)
        faster one starts to overfit -- the ``k >= 2`` flow then carries a
        heavy importance-weight tail.  Training each expert as its own
        single group-mixture flow on its cluster's points (the
        ``n_clusters_max == 1`` code path) removes that coupling: each
        expert gets its own optimiser and runs to its own validation-loss
        early stop.
        """
        model = self.model
        if not (
            isinstance(model, ClusteredGroupMixtureFlowWrapper)
            and model._n_active_experts() >= 2
            and weights is None
            and conditional is None
        ):
            return super().train(
                samples, weights=weights, conditional=conditional, plot=plot,
                **kwargs,
            )

        if not self.initialised:
            self.initialise()
        if not np.isfinite(samples).all():
            raise ValueError("Training data is not finite")

        output = kwargs.pop("output", None) or self.output
        os.makedirs(output, exist_ok=True)
        labels = model.route_prime_array(samples)
        k = model._n_active_experts()

        full_model, full_opt = self.model, self._optimiser
        history = dict(loss=[], val_loss=[])
        try:
            for j in range(k):
                sub = np.ascontiguousarray(samples[labels == j])
                if sub.shape[0] < 2:
                    logger.warning(
                        "Clustered group mixture: expert %d has %d routed "
                        "points -- skipping its training this round",
                        j, sub.shape[0],
                    )
                    continue
                self.model = model.experts[j]
                self._optimiser = self.get_optimiser()
                hj = super().train(
                    sub, plot=False,
                    output=os.path.join(output, f"expert_{j}"), **kwargs,
                )
                history["loss"].append(hj["loss"])
                history["val_loss"].append(hj["val_loss"])
                logger.info(
                    "Clustered group mixture: expert %d solo-trained on %d "
                    "pts (%d epochs, best val loss %.4g)",
                    j, sub.shape[0], len(hj["loss"]),
                    min(hj["val_loss"]) if hj["val_loss"] else float("nan"),
                )
            if model._bg_expert is not None and samples.shape[0] >= 2:
                self.model = model._bg_expert
                self._optimiser = self.get_optimiser()
                hb = super().train(
                    np.ascontiguousarray(samples), plot=False,
                    output=os.path.join(output, "expert_bg"), **kwargs,
                )
                history["loss"].append(hb["loss"])
                history["val_loss"].append(hb["val_loss"])
                model._bg_seen.fill_(True)
                logger.info(
                    "Clustered group mixture: background expert trained on "
                    "%d pts (%d epochs, best val loss %.4g, blend weight %.3g)",
                    samples.shape[0], len(hb["loss"]),
                    min(hb["val_loss"]) if hb["val_loss"] else float("nan"),
                    float(model._bg_weight),
                )
        finally:
            self.model, self._optimiser = full_model, full_opt

        self.model.train()
        self.model.eval()
        self.finalise()
        self.save_weights(os.path.join(output, "model.pt"))
        self.move_to(self.inference_device)
        self.model.eval()
        return history

    def get_model(self, config):
        k = max(int(getattr(self, "n_clusters_max", 1)), 1)
        experts = [
            GroupMixtureFlowModel.get_model(self, config) for _ in range(k)
        ]
        if k == 1:
            return experts[0]
        bg_weight = float(getattr(self, "bg_weight", 0.0))
        bg_expert = (
            GroupMixtureFlowModel.get_model(self, config)
            if bg_weight > 0.0
            else None
        )
        return ClusteredGroupMixtureFlowWrapper(
            experts,
            experts[0].num_features,
            cluster_method=getattr(self, "cluster_method", "gmm"),
            max_cluster_overlap=getattr(self, "max_cluster_overlap", 0.05),
            min_cluster_size=getattr(self, "min_cluster_size", 200),
            k_shrink_patience=getattr(self, "k_shrink_patience", 3),
            k_grow_patience=getattr(self, "k_grow_patience", 2),
            centroid_ema=getattr(self, "centroid_ema", None),
            bg_expert=bg_expert,
            bg_weight=bg_weight,
        )


def make_clustered_group_mixture_flow(
    *,
    n_clusters_max=2,
    cluster_method="gmm",
    max_cluster_overlap=0.05,
    min_cluster_size=200,
    k_shrink_patience=3,
    k_grow_patience=2,
    centroid_ema=None,
    bg_weight=0.0,
    **kwargs,
):
    """:func:`make_group_mixture_flow` with a clustered base flow.

    ``n_clusters_max`` independent group-mixture experts, one per data cluster
    found each training round (GMM; ``k`` = the largest value in
    ``[1, n_clusters_max]`` whose components are all well separated -- at most
    ``max_cluster_overlap`` of points ambiguous).  ``k == 1`` -- the data is
    not decisively multi-modal, or ``n_clusters_max == 1`` -- is byte-identical
    to :func:`make_group_mixture_flow` (``get_model`` returns the plain
    wrapper).  ``k`` evolves over the run -- it rises once the folded posterior
    becomes decisively multi-modal (the new expert is warm-started from the
    first) and falls again if it collapses back, with hysteresis both ways
    (``k_grow_patience`` / ``k_shrink_patience`` consecutive rounds).  The
    clustering is warm-started from the previous round and the cluster<->expert
    identity is held by centroid matching + ``centroid_ema`` smoothing, so
    routing does not jitter round to round.  All other keyword arguments are
    passed straight through.
    """
    base_cls = make_group_mixture_flow(**kwargs)

    class ClusteredCustom(ClusteredGroupMixtureFlowModel, base_cls):
        pass

    ClusteredCustom.n_clusters_max = int(n_clusters_max)
    ClusteredCustom.cluster_method = cluster_method
    ClusteredCustom.max_cluster_overlap = float(max_cluster_overlap)
    ClusteredCustom.min_cluster_size = int(min_cluster_size)
    ClusteredCustom.k_shrink_patience = int(k_shrink_patience)
    ClusteredCustom.k_grow_patience = int(k_grow_patience)
    ClusteredCustom.centroid_ema = (
        None if centroid_ema is None else float(centroid_ema)
    )
    ClusteredCustom.bg_weight = float(bg_weight)
    return ClusteredCustom


class GroupFlowProposalMixin:
    """Mixin wiring a :class:`~nessai.proposal.FlowProposal` to a group-mixture flow.

    Keeps the group-mixture flow's ``group_action_fn`` /
    ``in_fundamental_domain`` in the physical parameter space by wiring the
    proposal's reparameterisation into the flow as a
    :class:`CoordinateBridge`. The bridge is refreshed whenever the
    reparameterisation updates (e.g. data-driven z-score bounds). An affine
    reparameterisation gets a free :class:`AffineBridge`; anything else gets
    a :class:`ReparamBridge`.
    """

    def _structured(self, array, names):
        out = empty_structured_array(len(array), names=list(names))
        for i, name in enumerate(names):
            out[name] = array[:, i]
        return out

    def _refresh_group_affine_map(self):
        """Backwards-compatible alias for :meth:`_refresh_group_coordinate_bridge`."""
        self._refresh_group_coordinate_bridge()

    def _refresh_group_coordinate_bridge(self):
        """Recover the prime<->physical map and hand it to the flow.

        For an affine reparameterisation ``physical = prime * scale + shift``;
        ``scale`` and ``shift`` are recovered by probing ``inverse_rescale``
        and an :class:`AffineBridge` installed. A non-affine (or
        dimension-changing) reparameterisation gets a :class:`ReparamBridge`
        that round-trips through ``rescale`` / ``inverse_rescale``.
        """
        flow_model = getattr(self.flow, "model", None)
        if flow_model is None or not hasattr(
            flow_model, "set_coordinate_bridge"
        ):
            return
        if getattr(flow_model, "uses_prime_space_action", False):
            return
        prime = list(self.prime_parameters)
        physical = list(self.parameters)
        dtype = flow_model.weights.dtype
        device = flow_model.weights.device

        if len(prime) == len(physical):
            d = len(prime)

            def probe(value):
                arr = np.full((1, d), value, dtype=float)
                x, _ = self.inverse_rescale(self._structured(arr, prime))
                return np.array([x[name][0] for name in physical])

            p0, p1, phalf = probe(0.0), probe(1.0), probe(0.5)
            scale = p1 - p0
            shift = p0
            if np.allclose(
                phalf, shift + 0.5 * scale, rtol=1e-4, atol=1e-6
            ):
                flow_model.set_coordinate_bridge(
                    AffineBridge(
                        torch.as_tensor(scale, dtype=dtype),
                        torch.as_tensor(shift, dtype=dtype),
                    )
                )
                return

        flow_model.set_coordinate_bridge(
            ReparamBridge(
                prime_names=prime,
                physical_names=physical,
                rescale_fn=lambda s: self.rescale(s),
                inverse_rescale_fn=lambda s: self.inverse_rescale(s),
                dtype=dtype,
                device=device,
            )
        )

    def _training_data_as_prime_tensor(self, x):
        x_prime, _ = self.rescale(x.copy())
        arr = live_points_to_array(x_prime, self.prime_parameters, copy=True)
        model = self.flow.model
        return torch.as_tensor(
            arr, dtype=model.weights.dtype, device=model.weights.device
        )

    def check_state(self, x):
        super().check_state(x)
        self._refresh_group_coordinate_bridge()
        model = getattr(self.flow, "model", None)
        if model is None or not hasattr(model, "update_mixture_weights"):
            return
        # Run before the flow trains so the weights/standardisation reflect
        # the current data: the base flow is then never asked to score a
        # point that sits in a (currently) zero-weight element.
        x_prime = self._training_data_as_prime_tensor(x)
        model.update_mixture_weights(x_prime)
        model.update_base_standardisation(x_prime)
        self._log_group_weight_entropy()

    def _log_group_weight_entropy(self):
        flow_model = getattr(self.flow, "model", None)
        p = getattr(flow_model, "weights", None)
        if p is None:
            return

        frac = getattr(flow_model, "_last_leakage_fraction", None)
        frac_domain = getattr(flow_model, "_last_leakage_fraction_domain", 0.0)
        frac_rt = getattr(flow_model, "_last_leakage_fraction_roundtrip", 0.0)
        if frac is not None and frac > 0.1:
            logger.warning(
                "Group-mixture leakage fraction: %.3f (out-of-domain %.3f, "
                "non-round-trip %.3f) -- >0.1: the base flow is placing mass "
                "outside the canonical fundamental domain (out-of-domain: "
                "past the prior box; non-round-trip: a non-injective action "
                "that clamps outside it, e.g. raw angles through asin/clamp), "
                "which inflates the importance-weight spread and depresses "
                "the population acceptance.",
                frac,
                frac_domain,
                frac_rt,
            )
        elif frac is not None:
            logger.info(
                "Group-mixture leakage fraction: %.3f (out-of-domain %.3f, "
                "non-round-trip %.3f)",
                frac,
                frac_domain,
                frac_rt,
            )

        if not logger.isEnabledFor(logging.INFO):
            return
        p = p.detach()
        entropy = float(-(p * torch.log2(p.clamp_min(1e-12))).sum())
        n = p.numel()
        weights = np.array2string(
            p.cpu().numpy(), precision=3, separator=", ", suppress_small=True
        )
        logger.info(
            f"Group-mixture weight entropy: {entropy:.3f} bits "
            f"({entropy / np.log2(n):.3f} normalised); weights: {weights}"
        )

        # Factorised estimator: also report each generator's marginal, so a
        # collapsing factor is visible before it drags a whole slice of joint
        # weights to zero.
        marginals = getattr(flow_model, "_last_factor_marginals", None)
        counts = getattr(flow_model, "_last_factor_counts", None)
        sizes = getattr(flow_model, "mode_factor_sizes", None)
        if marginals is not None and sizes is not None:
            patience = getattr(flow_model, "_weight_empty_patience", 3)
            empty = getattr(flow_model, "_factor_empty_rounds", None)
            offsets, off = [], 0
            for s in sizes:
                offsets.append(off)
                off += s
            n_assigned = (
                int(counts[0].sum()) if counts is not None else None
            )
            if n_assigned is not None:
                logger.info(
                    "  factorised weights from %d assigned live points "
                    "(each generator's counts partition this same set):",
                    n_assigned,
                )
            for f, (size, m) in enumerate(zip(sizes, marginals)):
                m = m.numpy()
                ent = float(-(m * np.log2(np.clip(m, 1e-12, None))).sum())
                c = (
                    counts[f].numpy().astype(int).tolist()
                    if counts is not None
                    else None
                )
                dropped = []
                if empty is not None:
                    er = empty[offsets[f] : offsets[f] + size]
                    dropped = (er >= patience).nonzero(as_tuple=True)[0]
                    dropped = dropped.cpu().tolist()
                logger.info(
                    "  generator %d (Z%d): marginal %s%s%s "
                    "[entropy %.3f / %.3f bits]",
                    f,
                    size,
                    np.array2string(
                        m, precision=3, separator=", ", suppress_small=True
                    ),
                    f", counts {c}" if c is not None else "",
                    f", dropped values {dropped}" if dropped else "",
                    ent,
                    np.log2(size),
                )


class GroupFlowProposal(GroupFlowProposalMixin, FlowProposal):
    """:class:`~nessai.proposal.FlowProposal` wired for a group-mixture flow."""
