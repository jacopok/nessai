"""
Discrete Group Mixture Flow extension for nessai using Python Mixins.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class DiscreteGroupMixtureFlowWrapper(nn.Module):
    """
    PyTorch wrapper that applies discrete group transformations to a base normalizing flow.

    Computes:
        log p(x) = LogSumExp_g [ log p_base(g^-1 * x) + log pi_g ]
    where pi_g are learnable constant mixture weights (logits) for each group element.
    """
    def __init__(self, base_flow, num_features, group_action_fn, group_size, param_names=None):
        super().__init__()
        self.base_flow = base_flow
        self.num_features = num_features
        self.group_size = group_size
        self.group_action_fn = group_action_fn
        self.param_names = param_names or [f"p_{i}" for i in range(num_features)]

        # Learnable global constant logits vector (shape: [group_size])
        self.logits = nn.Parameter(torch.zeros(group_size))

    def _apply_group_action(self, z_flat: torch.Tensor, modes_flat: torch.Tensor, inverse: bool) -> torch.Tensor:
        point_dict = {name: z_flat[:, i] for i, name in enumerate(self.param_names)}
        mapped_dict = self.group_action_fn(point_dict, modes_flat, inverse=inverse)
        return torch.stack([mapped_dict[name] for name in self.param_names], dim=-1)

    def log_prob(self, x, context=None):
        batch_size = x.shape[0]

        # Expand inputs across all |G| group modes
        x_expanded = x.unsqueeze(0).repeat(self.group_size, 1, 1)
        modes = torch.arange(self.group_size, device=x.device).unsqueeze(1).repeat(1, batch_size)

        x_flat = x_expanded.view(self.group_size * batch_size, -1)
        modes_flat = modes.view(self.group_size * batch_size)

        # Evaluate g^-1 * x in base flow
        z_prime_flat = self._apply_group_action(x_flat, modes_flat, inverse=True)
        base_lp_flat = self.base_flow.log_prob(z_prime_flat, context=context)

        # Expand constant logits across batch
        log_pi = torch.log_softmax(self.logits, dim=-1)
        log_pi_expanded = log_pi.unsqueeze(1).repeat(1, batch_size).view(-1)

        # LogSumExp over discrete branches
        branch_log_probs = (base_lp_flat + log_pi_expanded).view(self.group_size, batch_size).T
        return torch.logsumexp(branch_log_probs, dim=-1)

    def sample_and_log_prob(self, num_samples, context=None):
        z_prime, _ = self.base_flow.sample_and_log_prob(num_samples, context=context)

        # Draw modes according to constant categorical distribution
        dist = Categorical(logits=self.logits)
        modes = dist.sample((num_samples,))

        # Map base samples via forward group action g * z'
        x = self._apply_group_action(z_prime, modes, inverse=False)
        exact_log_prob = self.log_prob(x, context=context)
        return x, exact_log_prob


class DiscreteGroupMixtureMixin:
    """
    Mixin class for nessai FlowModel architectures.
    Intercepts get_model() to wrap any built-in base flow with group mixture logic.
    """
    group_action_fn = None
    group_size = None
    param_names = None

    def get_model(self, config, **kwargs):
        base_flow = super().get_model(config, **kwargs)

        group_action_fn = getattr(self, 'group_action_fn', config.get('group_action_fn'))
        group_size = getattr(self, 'group_size', config.get('group_size'))
        param_names = getattr(self, 'param_names', config.get('param_names'))
        num_features = config.get('n_inputs')

        if group_action_fn is None or group_size is None:
            raise ValueError("DiscreteGroupMixtureMixin requires `group_action_fn` and `group_size`.")

        return DiscreteGroupMixtureFlowWrapper(
            base_flow=base_flow,
            num_features=num_features,
            group_action_fn=group_action_fn,
            group_size=group_size,
            param_names=param_names
        )


def make_group_mixture_flow(base_flow_class, group_action_fn, group_size, param_names):
    """
    Factory creating a dynamic Mixin FlowModel class combining DiscreteGroupMixtureMixin
    with any existing nessai flow architecture (e.g., ResNetFlowModel, StandardFlowModel).
    """
    class GroupMixtureFlowClass(DiscreteGroupMixtureMixin, base_flow_class):
        pass

    GroupMixtureFlowClass.group_action_fn = staticmethod(group_action_fn)
    GroupMixtureFlowClass.group_size = group_size
    GroupMixtureFlowClass.param_names = param_names
    return GroupMixtureFlowClass