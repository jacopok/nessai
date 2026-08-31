"""
Discrete group-mixture flow extension for nessai.
"""

import logging

import numpy as np
import torch
from torch.distributions import Categorical

from ..flows.base import BaseFlow
from ..flows.utils import configure_model
from ..livepoint import empty_structured_array, live_points_to_array
from ..proposal import FlowProposal
from .base import FlowModel

logger = logging.getLogger(__name__)


class DiscreteGroupMixtureFlowWrapper(BaseFlow):
    """Wrap a base flow with a discrete group-mixture transformation.

    Computes ``log p(x) = logsumexp_g [log p_base(g^-1 x) + log pi_g]``. The
    mixture weights ``pi_g`` are not trained by gradient descent: the
    fundamental-domain mask assigns each point to exactly one group element,
    so the maximum-likelihood weights are the assigned-point fractions, set
    in closed form by :meth:`update_mixture_weights`.
    """

    def __init__(
        self,
        base_flow,
        num_features,
        group_action_fn,
        group_size,
        param_names=None,
        in_fundamental_domain=None,
    ):
        super().__init__()
        self.base_flow = base_flow
        self.num_features = num_features
        self.group_size = group_size
        self.group_action_fn = group_action_fn
        self.in_fundamental_domain = in_fundamental_domain
        self.param_names = param_names or [
            f"p_{i}" for i in range(num_features)
        ]

        # Affine prime->physical map ``physical = prime * scale + shift``,
        # filled each round by ``GroupFlowProposalMixin`` from the proposal's
        # reparameterisation. The user's ``group_action_fn`` /
        # ``in_fundamental_domain`` are written in physical coordinates.
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

    def _to_physical(self, z):
        return z * self._prime_scale + self._prime_shift

    def _to_prime(self, x):
        return (x - self._prime_shift) / self._prime_scale

    def _standardise(self, canon, modes):
        return (canon - self._canon_mean[modes]) / self._canon_std[modes]

    def _destandardise(self, u, modes):
        return u * self._canon_std[modes] + self._canon_mean[modes]

    def _canon_log_det(self, modes):
        return -torch.log(self._canon_std[modes]).sum(-1)

    def _preimages(self, x):
        """All group pre-images ``g_k^-1 . x`` in prime coords: ``[K, B, d]``."""
        k, b = self.group_size, x.shape[0]
        x_rep = x.unsqueeze(0).expand(k, b, -1).reshape(k * b, -1)
        modes = torch.arange(k, device=x.device).repeat_interleave(b)
        return self._apply_group_action(x_rep, modes, inverse=True).view(
            k, b, -1
        )

    def _assign_branch(self, x):
        """Geometric branch assignment (no base-flow evaluation).

        Returns ``(assigned [B], preimages [K, B, d], claimed [B])`` where
        ``assigned`` is the group element whose inverse maps ``x`` into the
        fundamental domain (0 for points no element claims).
        """
        pre = self._preimages(x)
        if self.in_fundamental_domain is None:
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
        gmean = canon.mean(dim=0)
        gstd = canon.std(dim=0).clamp_min(1e-6)
        beta = self._canon_ema
        for k in range(self.group_size):
            sel = claimed & (assigned == k)
            if int(sel.sum()) >= self._min_std_count:
                c = canon[sel]
                mk = c.mean(dim=0)
                sk = c.std(dim=0).clamp_min(1e-6)
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

    def _apply_group_action(
        self, z_flat: torch.Tensor, modes_flat: torch.Tensor, inverse: bool
    ) -> torch.Tensor:
        phys = self._to_physical(z_flat)
        point_dict = {
            name: phys[:, i] for i, name in enumerate(self.param_names)
        }
        mapped_dict = self.group_action_fn(
            point_dict, modes_flat, inverse=inverse
        )
        out = torch.stack(
            [mapped_dict[name] for name in self.param_names], dim=-1
        )
        return self._to_prime(out)

    def _in_domain(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Boolean mask: which rows of ``z_flat`` lie in the fundamental domain."""
        phys = self._to_physical(z_flat)
        point_dict = {
            name: phys[:, i] for i, name in enumerate(self.param_names)
        }
        return self.in_fundamental_domain(point_dict).to(torch.bool)

    def _branch_log_probs(self, x, context=None):
        """Return ``base_lp(g_k^-1 x) + log pi_k`` for every group element: ``[K, B]``.

        Branches whose pre-image is outside the fundamental domain are
        ``-inf`` and the base flow is not evaluated for them, so for a group
        that tiles the space this costs a single base-flow call on ``B``
        canonical points rather than ``K * B``.
        """
        k, b = self.group_size, x.shape[0]
        pre = self._preimages(x)
        log_pi = torch.log(self.weights).unsqueeze(1).expand(k, b)
        flat_pre = pre.reshape(k * b, -1)
        flat_modes = torch.arange(k, device=x.device).repeat_interleave(b)

        if self.in_fundamental_domain is None:
            base_lp = (
                self.base_flow.log_prob(
                    self._standardise(flat_pre, flat_modes), context=context
                )
                + self._canon_log_det(flat_modes)
            ).view(k, b)
            return base_lp + log_pi

        in_dom = self._in_domain(flat_pre).view(k, b)
        # Points no branch claims: evaluate all their branches (fallback).
        eval_mask = in_dom | (~in_dom.any(dim=0, keepdim=True))
        idx = eval_mask.reshape(-1).nonzero(as_tuple=True)[0]
        base_lp = flat_pre.new_full((k * b,), -float("inf"))
        if idx.numel():
            base_lp[idx] = self.base_flow.log_prob(
                self._standardise(flat_pre[idx], flat_modes[idx]),
                context=context,
            ) + self._canon_log_det(flat_modes[idx])
        return base_lp.view(k, b) + log_pi

    def log_prob(self, x, context=None):
        return torch.logsumexp(
            self._branch_log_probs(x, context=context), dim=0
        )

    def _mixture_log_prob_from_canonical(
        self, canon, modes, x, log_q0_canon, context=None
    ):
        """log q(x) for x generated as ``g_modes . canon``.

        For a group that tiles the space and ``canon`` inside the fundamental
        domain only branch ``modes`` contributes, so ``log q(x) = log pi_modes
        + log q0(canon)`` with no extra base-flow evaluation. Points whose
        ``canon`` leaked out of the domain fall back to the full mixture.
        """
        log_q = torch.log(self.weights[modes]) + log_q0_canon
        if self.in_fundamental_domain is not None:
            bad = ~self._in_domain(canon)
            if bool(bad.any()):
                log_q = log_q.clone()
                log_q[bad] = self.log_prob(x[bad], context=context)
        return log_q

    def sample_and_log_prob(self, num_samples, context=None):
        u, log_q0_u = self.base_flow.sample_and_log_prob(
            num_samples, context=context
        )
        modes = Categorical(probs=self.weights).sample((num_samples,))
        canon = self._destandardise(u, modes)
        x = self._apply_group_action(canon, modes, inverse=False)
        log_q = self._mixture_log_prob_from_canonical(
            canon, modes, x, log_q0_u + self._canon_log_det(modes), context
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
        return self.base_flow.forward(
            self._standardise(canon, assigned), context=context
        )

    def inverse(self, z, context=None):
        # ``log_j`` is set so that
        # ``latent_log_prob(z) - log_j == self.log_prob(x)``, the quantity
        # ``FlowProposal.backward_pass`` relies on. It is not a literal
        # Jacobian: the group action is assumed measure-preserving.
        modes = Categorical(probs=self.weights).sample((z.shape[0],))

        u, inv_log_j = self.base_flow.inverse(z, context=context)
        canon = self._destandardise(u, modes)
        x = self._apply_group_action(canon, modes, inverse=False)

        latent_log_prob = self.base_flow.base_distribution_log_prob(
            z, context=context
        )
        log_q0_canon = latent_log_prob - inv_log_j + self._canon_log_det(modes)
        log_q = self._mixture_log_prob_from_canonical(
            canon, modes, x, log_q0_canon, context
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
        )


def make_group_mixture_flow(
    group_action_fn, group_size, param_names, in_fundamental_domain=None
):
    """Factory constructing a ``GroupMixtureFlowModel`` bound to a specific group.

    Parameters
    ----------
    group_action_fn : callable
        ``(point_dict, modes, inverse=False) -> point_dict`` applying the
        group action, in the flow's input coordinates.
    group_size : int
        Number of discrete group elements.
    param_names : list of str
        Ordered parameter names.
    in_fundamental_domain : callable, optional
        ``(point_dict) -> bool tensor`` marking whether each point is the
        canonical orbit representative, defined in the physical parameter
        space. If omitted the mixture weights are not identifiable (the
        base flow can absorb the whole distribution).

    Notes
    -----
    ``group_action_fn`` and ``in_fundamental_domain`` are defined in the
    physical parameter space, so the proposal class must use
    :class:`GroupFlowProposalMixin` unless the flow coordinates already
    equal the physical ones.
    """

    class CustomGroupMixtureFlowModel(GroupMixtureFlowModel):
        pass

    CustomGroupMixtureFlowModel.group_action_fn = staticmethod(group_action_fn)
    CustomGroupMixtureFlowModel.group_size = group_size
    CustomGroupMixtureFlowModel.param_names = param_names
    if in_fundamental_domain is not None:
        CustomGroupMixtureFlowModel.in_fundamental_domain = staticmethod(
            in_fundamental_domain
        )
    return CustomGroupMixtureFlowModel


class GroupFlowProposalMixin:
    """Mixin wiring a :class:`~nessai.proposal.FlowProposal` to a group-mixture flow.

    Keeps the group-mixture flow's ``group_action_fn`` /
    ``in_fundamental_domain`` in the physical parameter space by wiring the
    proposal's reparameterisation into the flow as prime<->physical
    coordinate maps. The maps are refreshed whenever the reparameterisation
    updates (e.g. data-driven z-score bounds).
    """

    def _structured(self, array, names):
        out = empty_structured_array(len(array), names=list(names))
        for i, name in enumerate(names):
            out[name] = array[:, i]
        return out

    def _refresh_group_affine_map(self):
        """Recover the affine prime->physical map and hand it to the flow.

        For an affine reparameterisation ``physical = prime * scale +
        shift``; ``scale`` and ``shift`` are recovered by probing
        ``inverse_rescale``. A non-affine reparameterisation raises.
        """
        flow_model = getattr(self.flow, "model", None)
        if flow_model is None or not hasattr(flow_model, "set_affine_maps"):
            return
        prime = list(self.prime_parameters)
        physical = list(self.parameters)
        d = len(prime)

        def probe(value):
            arr = np.full((1, d), value, dtype=float)
            x, _ = self.inverse_rescale(self._structured(arr, prime))
            return np.array([x[name][0] for name in physical])

        p0, p1, phalf = probe(0.0), probe(1.0), probe(0.5)
        scale = p1 - p0
        shift = p0
        if not np.allclose(phalf, shift + 0.5 * scale, rtol=1e-4, atol=1e-6):
            raise RuntimeError(
                "GroupFlowProposalMixin requires an affine reparameterisation "
                "(null, scale-and-shift or z-score); got a non-affine one."
            )
        dtype = flow_model.weights.dtype
        flow_model.set_affine_maps(
            torch.as_tensor(scale, dtype=dtype),
            torch.as_tensor(shift, dtype=dtype),
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
        self._refresh_group_affine_map()
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
        if p is None or not logger.isEnabledFor(logging.INFO):
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
