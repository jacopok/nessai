"""
2D test script utilizing nessai.flowmodel.group_mixture.
"""

import numpy as np
import torch
from nessai.model import Model
from nessai.flowsampler import FlowSampler
from nessai.proposal import FlowProposal

from nessai.flowmodel.group_mixture import make_group_mixture_flow
import logging
logging.basicConfig(level=logging.INFO)

N_PERIODS = 5

# =====================================================================
# 1. Group Action & Flow Definition
# =====================================================================
def shift_periodic_2d_group_action(point_dict: dict, modes_flat: torch.Tensor, inverse: bool = False) -> dict:
    x = point_dict['x']
    y = point_dict['y']
    shift = modes_flat.to(x.dtype)
    
    x_mapped = x - shift if inverse else x + shift
    return {'x': x_mapped, 'y': y}

def fold_parameter_space(point_dict: dict) -> dict:
    x = point_dict['x']
    y = point_dict['y']

    return {'x': x % 1, 'y': y}

RealNVPGroupFlow = make_group_mixture_flow(
    group_action_fn=shift_periodic_2d_group_action,
    group_size=N_PERIODS,
    param_names=['x', 'y'],
    fold_fn=fold_parameter_space,
)


class GroupFlowProposal(FlowProposal):
    """FlowProposal that uses the custom group-mixture flow model."""
    _FlowModelClass = RealNVPGroupFlow


# =====================================================================
# 2. 2D Target Model (N_PERIODS Modes in X, Unimodal in Y)
# =====================================================================
class Periodic2DModel(Model):
    def __init__(self):
        self.names = ['x', 'y']
        self.bounds = {'x': (0.0, float(N_PERIODS)), 'y': (-5.0, 5.0)}
        self.num_periods = N_PERIODS
        self.sigma_x = 0.02
        self.sigma_y = 1.0
        self.centers_x = np.arange(N_PERIODS) + 0.5

        raw_amps = np.array([0.5, 2.0, 1.2, 0.8, 3.5, 0.3, 1.8, 2.5, 0.9, 1.4])[:N_PERIODS]
        self.true_amplitudes = raw_amps / np.sum(raw_amps)

    def log_prior(self, x):
        in_b = self.in_bounds(x)
        log_p = np.full(x.size, -np.log(N_PERIODS * N_PERIODS))
        log_p[~in_b] = -np.inf
        return log_p

    def log_likelihood(self, struct_array):
        x_val = struct_array['x']
        y_val = struct_array['y']
        
        gauss_y = np.exp(-0.5 * (y_val / self.sigma_y) ** 2) / (np.sqrt(2 * np.pi) * self.sigma_y)
        
        gauss_x_mix = np.zeros_like(x_val)
        for k in range(self.num_periods):
            c = self.centers_x[k]
            w = self.true_amplitudes[k]
            g_x = np.exp(-0.5 * ((x_val - c) / self.sigma_x) ** 2) / (np.sqrt(2 * np.pi) * self.sigma_x)
            gauss_x_mix += w * g_x

        lik = gauss_x_mix * gauss_y
        return np.log(np.maximum(lik, 1e-300))


# =====================================================================
# 3. Sampling Pipeline
# =====================================================================
if __name__ == "__main__":
    target_model = Periodic2DModel()

    flow_config_group = {
        'model': 'realnvp',
        'n_blocks': 4,
        'n_neurons': 64
    }
    flow_config_regular = {
        'ftype': 'realnvp',
        'n_blocks': 4,
        'n_neurons': 64
    }

    print("=== Sampling with regular FlowProposal (baseline) ===")
    sampler_regular = FlowSampler(
        Periodic2DModel(),
        output="./outdir_2d_regular/",
        flow_proposal_class=FlowProposal,
        flow_config=flow_config_regular,
        nlive=1000,
        resume=False,
        seed=42,
    )
    sampler_regular.run()

    print("=== Sampling with RealNVPGroupFlow (nessai.flowmodel.group_mixture) ===")
    sampler_group = FlowSampler(
        Periodic2DModel(),
        output="./outdir_2d_group/",
        flow_proposal_class=GroupFlowProposal,
        flow_config=flow_config_group,
        nlive=1000,
        resume=False,
        seed=42,
    )
    sampler_group.run()

    # Extract wrapper module directly from trained proposal
    trained_wrapper = sampler_group.ns._flow_proposal.flow.model

    with torch.no_grad():
        learned_probs = torch.softmax(trained_wrapper.logits, dim=-1).cpu().numpy()

    print("\n" + "=" * 65)
    print(" AMPLITUDE COMPARISON: Learned Logit Weights vs Ground Truth")
    print("=" * 65)
    print(f"{'Period (k)':<12} | {'True Amplitude':<18} | {'Learned Logit Prob':<20}")
    print("-" * 65)
    for k in range(N_PERIODS):
        print(f"{k:<12} | {target_model.true_amplitudes[k]:<18.4f} | {learned_probs[k]:<20.4f}")
    print("-" * 65)