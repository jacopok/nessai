Group-mixture proposal
======================

The group-mixture proposal targets posteriors with a **known discrete
symmetry**: distributions that are invariant (or nearly so) under a finite
group :math:`G` acting on the parameter space. Typical examples are periodic
parameters, sign/reflection degeneracies, and label-switching between
exchangeable components.

Instead of asking a single normalising flow to learn every symmetric copy of
the distribution, the flow only learns one *fundamental domain* and the
proposal reconstructs the full posterior as a mixture over the group:

.. math::

   \log p(x) = \operatorname{logsumexp}_{g \in G}
   \left[ \log p_\text{base}(g^{-1} x) + \log \pi_g \right],

where the mixture weights :math:`\pi_g` are set in closed form each training
round to the fraction of live points assigned to each group element.

The implementation lives in :mod:`nessai.flowmodel.group_mixture`.


What you need to provide
------------------------

To use the proposal you supply four things, via
:func:`~nessai.flowmodel.group_mixture.make_group_mixture_flow`:

``param_names`` (list of str)
    The ordered names of the parameters the flow sees, matching
    ``model.names``. All callables below receive/return dictionaries keyed
    by these names.

``group_size`` (int)
    The number :math:`|G|` of discrete group elements, including the
    identity. For a parameter with ``N`` periods this is ``N``; for a single
    reflection it is ``2``.

``group_action_fn`` (callable)
    ``group_action_fn(point_dict, modes, inverse=False) -> point_dict``

    Applies the group action to a batch of points. ``point_dict`` maps each
    name in ``param_names`` to a 1-D tensor of length ``B``; ``modes`` is an
    integer tensor of length ``B`` giving, per point, which group element
    :math:`g_k` to apply (``0`` must be the identity). When
    ``inverse=True`` it must apply :math:`g_k^{-1}` instead. The function
    must be written with torch operations (it runs inside autograd) and
    must be vectorised over the batch.

``in_fundamental_domain`` (callable, optional but strongly recommended)
    ``in_fundamental_domain(point_dict) -> bool tensor``

    Returns, per point, whether it is the canonical orbit representative -
    i.e. whether it lies in the fundamental domain. The regions
    :math:`\{g\,\mathcal{D} : g \in G\}` should tile the prior support with
    no overlap and no gap. This mask is what makes the mixture weights
    identifiable and lets the proposal evaluate a single base-flow call per
    point instead of :math:`|G|`; if omitted, the base flow can absorb the
    whole distribution and the weights carry no meaning.

Both ``group_action_fn`` and ``in_fundamental_domain`` are defined in the
**physical parameter space** (the units of ``model.bounds``), not in the
flow's internal ``prime`` coordinates.


Requirements and assumptions
----------------------------

* **The group action is measure-preserving.** The action is assumed to have
  unit Jacobian (permutations, translations, reflections, rotations). Scaling
  actions are not supported.

* **The symmetry must be exact in the prior.** The prior density must be
  constant over each orbit (e.g. a uniform prior whose bounds respect the
  symmetry). If the prior is not symmetric the mixture weights will absorb
  the prior asymmetry as well as the likelihood's.

* **An affine reparameterisation.** Because the user callables are in
  physical coordinates, the proposal must reconcile them with the flow's
  ``prime`` coordinates. :class:`~nessai.flowmodel.group_mixture.GroupFlowProposalMixin`
  does this automatically, but only for an affine reparameterisation
  (``null``, ``scale``, ``zscore``/z-score, shift). A non-affine
  reparameterisation raises ``RuntimeError``. If your flow coordinates
  already equal the physical ones you can skip the mixin.

* **The fundamental domain should tile the space.** Overlap double-counts
  points; gaps leave points unclaimed (they fall back to the full,
  ``|G|``-times more expensive, mixture evaluation).


Wiring it into the sampler
--------------------------

Build the flow-model class, then combine the mixin with a
:class:`~nessai.proposal.FlowProposal`:

.. code-block:: python

    import torch
    from nessai.proposal import FlowProposal
    from nessai.flowsampler import FlowSampler
    from nessai.flowmodel.group_mixture import (
        make_group_mixture_flow,
        GroupFlowProposalMixin,
    )

    N_PERIODS = 10

    def shift_action(point_dict, modes, inverse=False):
        x = point_dict["x"]
        shift = modes.to(x.dtype)
        x = x - shift if inverse else x + shift
        return {"x": x, "y": point_dict["y"]}

    def in_fundamental_domain(point_dict):
        x = point_dict["x"]
        return (x >= 0.0) & (x < 1.0)

    GroupFlow = make_group_mixture_flow(
        group_action_fn=shift_action,
        group_size=N_PERIODS,
        param_names=["x", "y"],
        in_fundamental_domain=in_fundamental_domain,
    )

    class GroupFlowProposal(GroupFlowProposalMixin, FlowProposal):
        _FlowModelClass = GroupFlow

    sampler = FlowSampler(
        model,
        output="./outdir/",
        flow_proposal_class=GroupFlowProposal,
        flow_config={"model": "realnvp", "n_blocks": 4, "n_neurons": 64},
        nlive=2000,
    )
    sampler.run()

The learned weights are available after the run as
``sampler.ns._flow_proposal.flow.model.weights`` and estimate the relative
posterior mass of each symmetric copy.

.. note::

    A working end-to-end script is in ``test_periodic_group_mixture.py`` at
    the repository root.
