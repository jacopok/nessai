"""
Discrete group-mixture flow extension for nessai.
"""

import logging
import math

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
        """EM M-step: set the mixture weights to the assigned-point fractions.

        Each point is assigned to the single group element whose inverse
        action maps it into the fundamental domain, so the maximum-
        likelihood weights are the assignment counts. ``smoothing`` keeps a
        transiently-empty element in play; an element empty for
        ``_weight_empty_patience`` consecutive rounds is dropped (weight 0)
        until points return to it.
        """
        assigned, _, claimed = self._assign_branch(x)
        counts = torch.bincount(
            assigned[claimed], minlength=self.group_size
        ).to(self.weights)
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


class GroupFlowProposal(GroupFlowProposalMixin, FlowProposal):
    """:class:`~nessai.proposal.FlowProposal` wired for a group-mixture flow."""
