"""
nessai/flowmodel/group_mixture.py
Discrete Group Mixture Flow extension for nessai using Python Mixins.
"""

import logging

import torch
import torch.nn as nn
from torch.distributions import Categorical
from nessai.flowmodel import FlowModel
from nessai.flows.base import BaseFlow
from nessai.flows.utils import configure_model

logger = logging.getLogger(__name__)


class DiscreteGroupMixtureFlowWrapper(BaseFlow):
    """
    PyTorch wrapper that applies discrete group transformations to a base normalizing flow.

    Computes:
        log p(x) = LogSumExp_g [ log p_base(g^-1 * x) + log pi_g ]
    where pi_g are learnable constant mixture weights (logits) for each group element.
    """
    def __init__(self, base_flow, num_features, group_action_fn, group_size, param_names=None, fold_fn=None):
        super().__init__()
        self.base_flow = base_flow
        self.num_features = num_features
        self.group_size = group_size
        self.group_action_fn = group_action_fn
        self.fold_fn = fold_fn
        self.param_names = param_names or [f"p_{i}" for i in range(num_features)]

        # Learnable global constant logits vector (shape: [group_size])
        self.logits = nn.Parameter(torch.zeros(group_size))

    def _apply_group_action(self, z_flat: torch.Tensor, modes_flat: torch.Tensor, inverse: bool) -> torch.Tensor:
        point_dict = {name: z_flat[:, i] for i, name in enumerate(self.param_names)}
        mapped_dict = self.group_action_fn(point_dict, modes_flat, inverse=inverse)
        return torch.stack([mapped_dict[name] for name in self.param_names], dim=-1)

    def _fold(self, z_flat: torch.Tensor) -> torch.Tensor:
        if self.fold_fn is None:
            return z_flat
        point_dict = {name: z_flat[:, i] for i, name in enumerate(self.param_names)}
        mapped_dict = self.fold_fn(point_dict)
        return torch.stack([mapped_dict[name] for name in self.param_names], dim=-1)

    def _branch_log_probs(self, x, context=None):
        """Return ``base_lp(g^-1 x) + log pi_g`` for every group element.

        Shape ``[group_size, batch_size]``. A branch whose pre-image
        ``g^-1 x`` does not lie in the fundamental domain is set to
        ``-inf``: for a group that tiles the space exactly one branch per
        point survives, so ``logsumexp`` collapses to that branch and the
        mixture logit for it receives a responsibility-weighted gradient.
        The base flow is only ever evaluated on in-domain points, so it
        stays specialised to a single orbit representative.
        """
        batch_size = x.shape[0]
        x_expanded = x.unsqueeze(0).repeat(self.group_size, 1, 1)
        modes = torch.arange(self.group_size, device=x.device).unsqueeze(1).repeat(1, batch_size)

        x_flat = x_expanded.view(self.group_size * batch_size, -1)
        modes_flat = modes.view(self.group_size * batch_size)

        preimage_flat = self._apply_group_action(x_flat, modes_flat, inverse=True)
        folded_flat = self._fold(preimage_flat)
        # Fold before the base flow only for numerical safety (keeps far
        # out-of-domain branches finite); those branches are masked below.
        base_lp_flat = self.base_flow.log_prob(folded_flat, context=context)

        log_pi = torch.log_softmax(self.logits, dim=-1)
        log_pi_expanded = log_pi.unsqueeze(1).repeat(1, batch_size).view(-1)

        branch = (base_lp_flat + log_pi_expanded).view(self.group_size, batch_size)

        if self.fold_fn is not None:
            in_domain = torch.isclose(
                folded_flat, preimage_flat, atol=1e-4
            ).all(dim=-1).view(self.group_size, batch_size)
            # Guard against a point that no branch claims (domain gaps).
            in_domain = in_domain | (~in_domain.any(dim=0, keepdim=True))
            branch = branch.masked_fill(~in_domain, float("-inf"))
        return branch

    def log_prob(self, x, context=None):
        return torch.logsumexp(self._branch_log_probs(x, context=context), dim=0)

    def sample_and_log_prob(self, num_samples, context=None):
        z_prime, _ = self.base_flow.sample_and_log_prob(num_samples, context=context)

        # Draw modes according to constant categorical distribution
        dist = Categorical(logits=self.logits)
        modes = dist.sample((num_samples,))

        # Map base samples via forward group action g * z'
        x = self._apply_group_action(z_prime, modes, inverse=False)
        exact_log_prob = self.log_prob(x, context=context)
        return x, exact_log_prob

    # -- Remaining BaseFlow abstract methods -----------------------------

    def _map_to_canonical(self, x: torch.Tensor, context=None) -> torch.Tensor:
        """Find the most probable group element for each point and undo it.

        Used by ``forward``/``forward_and_log_prob``, e.g. to map a real
        data point into the latent space (for the truncation radius, or
        training diagnostics). Picks the group element that maximises the
        (mixture-weighted) branch log-probability for each point, then
        applies the inverse group action for that element.
        """
        best_modes = self._branch_log_probs(x, context=context).argmax(dim=0)
        return self._fold(self._apply_group_action(x, best_modes, inverse=True))

    def forward(self, x, context=None):
        x_prime = self._map_to_canonical(x, context=context)
        return self.base_flow.forward(x_prime, context=context)

    def inverse(self, z, context=None):
        # Sample which group element each point belongs to (matching the
        # mixture weights used in log_prob/sample_and_log_prob) and map the
        # base flow's output through the forward group action. `log_j` is
        # set so that `latent_log_prob(z) - log_j == self.log_prob(x)`,
        # which is the quantity FlowProposal.backward_pass relies on to
        # obtain the correct (mixture) proposal density for `x` -- it is
        # not a literal Jacobian since the group action is assumed to be
        # measure-preserving (e.g. a translation).
        batch_size = z.shape[0]
        dist = Categorical(logits=self.logits)
        modes = dist.sample((batch_size,))

        x_prime, _ = self.base_flow.inverse(z, context=context)
        x = self._apply_group_action(x_prime, modes, inverse=False)

        latent_log_prob = self.base_flow.base_distribution_log_prob(z, context=context)
        log_j = latent_log_prob - self.log_prob(x, context=context)
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
    """
    Custom FlowModel inheriting directly from nessai.flowmodel.FlowModel.
    Sanitizes model_config before building PyTorch flow models via configure_model.
    """
    group_action_fn = None
    group_size = None
    param_names = None
    fold_fn = None

    def initialise(self):
        """Initialise the model and optimiser.

        Overrides :meth:`~nessai.flowmodel.base.FlowModel.initialise` to
        build the model via :meth:`get_model` instead of calling
        :func:`~nessai.flows.utils.configure_model` directly, since the
        latter does not know how to handle the group-mixture-specific
        configuration keys.
        """
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

    def get_optimiser(self, optimiser=None, **kwargs):
        """Build the optimiser but keep the mixture logits out of weight decay.

        AdamW's weight decay would otherwise pull ``logits`` towards zero,
        i.e. the mixture towards a uniform distribution over group
        elements, which is rarely what the data supports.
        """
        optimiser = optimiser or self.optimiser
        default_kwargs = {
            "adam": {"weight_decay": 1e-6},
            "adamw": {},
            "sgd": {},
        }[optimiser.lower()]
        default_kwargs["lr"] = self.training_config["lr"]
        default_kwargs.update(self.optimiser_kwargs)
        default_kwargs.update(kwargs)

        optim_cls = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }[optimiser.lower()]

        logit_params = {id(self.model.logits)}
        decay = [p for p in self.model.parameters() if id(p) not in logit_params]
        no_decay = [self.model.logits]
        return optim_cls(
            [
                {"params": decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            **default_kwargs,
        )

    def get_model(self, config):
        # Work on a shallow copy to prevent modifying configuration dictionaries in-place
        config_clean = config.copy()

        # Remove keys that PyTorch flow constructors (e.g. RealNVP) do not expect
        config_clean.pop("model", None)
        group_action_fn = config_clean.pop("group_action_fn", getattr(self, "group_action_fn", None))
        group_size = config_clean.pop("group_size", getattr(self, "group_size", None))
        param_names = config_clean.pop("param_names", getattr(self, "param_names", None))
        fold_fn = config_clean.pop("fold_fn", getattr(self, "fold_fn", None))

        if group_action_fn is None or group_size is None:
            raise ValueError("GroupMixtureFlowModel requires `group_action_fn` and `group_size`.")

        # Construct standard underlying flow (RealNVP / NSF / MAF)
        base_flow = configure_model(config_clean)

        num_features = config_clean.get("n_inputs")

        # Wrap standard PyTorch flow into DiscreteGroupMixtureFlowWrapper
        return DiscreteGroupMixtureFlowWrapper(
            base_flow=base_flow,
            num_features=num_features,
            group_action_fn=group_action_fn,
            group_size=group_size,
            param_names=param_names,
            fold_fn=fold_fn,
        )


def make_group_mixture_flow(group_action_fn, group_size, param_names, fold_fn=None):
    """
    Factory constructing a customized GroupMixtureFlowModel class bound to specific group properties.
    """
    class CustomGroupMixtureFlowModel(GroupMixtureFlowModel):
        pass

    CustomGroupMixtureFlowModel.group_action_fn = staticmethod(group_action_fn)
    CustomGroupMixtureFlowModel.group_size = group_size
    CustomGroupMixtureFlowModel.param_names = param_names
    if fold_fn is not None:
        CustomGroupMixtureFlowModel.fold_fn = staticmethod(fold_fn)
    return CustomGroupMixtureFlowModel