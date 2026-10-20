==============================
Partitioned nested sampling
==============================

A flow trained on the live points can under-cover a small, separated group
of them: a secondary mode, or a diffuse tail. With a weight cap taken from a
quantile of the live points (the prior-relative ``log_proposal_threshold``
truncation), those points then set the cap, and the acceptance can fall by
orders of magnitude while they live. Partitioned nested sampling avoids this
without dropping anything: once the live points separate along a few
coordinates, the prior is split into frozen regions and each region is
continued as an independent nested sampling run, in lockstep with the
others.

How it works
============

- **The split.** Every ``check_every`` iterations a *splitter* looks at the
  live points along the *split coordinates* and may propose a *cut*: a fixed
  partition of those coordinates (cut points along one coordinate, or a
  line in a plane). The cut is frozen. A region is the set of parameters on
  one side of every cut made on the way to it, a fixed function of the
  parameters for the rest of the run.
- **The held-out half.** Each live point carries a fixed random bit, a hash
  of its log-likelihood and its iteration of birth. The splitter sees only
  the points with bit 0. The share of the prior volume of each part is the
  fraction of the points with bit 1 that fall in it, so it is not biased by
  the choice of the moment of the split.
- **The regions.** Each part keeps its live points, topped up with
  independent draws to ``n_min`` if it has fewer, and gets its own copy of
  the flow proposal. The copy is restricted to the region (draws outside it
  are discarded before the likelihood is evaluated), trained on the region's
  live points and retrained by its own retrain decision. Its weight cap
  comes from its own live points.
- **Lockstep.** The lowest live point among all active regions is killed,
  only its region's prior volume shrinks (by one over that region's number
  of live points), and it is replaced from its region's proposal. The
  likelihood threshold is common to all regions.
- **Stopping.** A region stops once it could add less than the stopping
  tolerance to the evidence (``ln(1 + X_r max L_r / Z)``); its live points
  are then added as the final points of a run. The run ends when every
  region has stopped. The dead points of all regions are merged in
  likelihood order; the evidence and the posterior weights are the sum over
  them of the likelihood times each point's own prior volume.

Any frozen partition gives an unbiased evidence; the choice of the
coordinates and of the cut affects only the efficiency, through how well
each region's flow covers its region.

Usage
=====

.. code-block:: python

    from nessai.flowsampler import FlowSampler

    fs = FlowSampler(
        model,
        output="outdir",
        nlive=2000,
        partition={
            # the coordinates the multimodality lies along
            "coordinates": ["x"],
            # optional: the splitter (default GapSplitter)
            "splitter": {
                "class": "nessai.partition:FirstOf",
                "splitters": [
                    {"class": "GapSplitter", "max_share": 0.25},
                    {"class": "CoreSplitter", "max_share": 0.25},
                ],
            },
        },
    )
    fs.run()

The options (:py:meth:`nessai.samplers.nestedsampler.NestedSampler.configure_partition`):

- ``coordinates`` (required): a list of parameter names, or a
  :py:class:`nessai.partition.SplitCoordinates`. Coordinates and splitters
  can be given as instances, as ``"module:Class"`` strings or as
  dictionaries with the class under ``"class"`` and its keyword arguments,
  which keeps the setting JSON-able.
- ``splitter``: :py:class:`nessai.partition.GapSplitter` (default: arcs or
  intervals separated by empty gaps, wider than ``ln n + 12`` local
  spacings), :py:class:`nessai.partition.CoreSplitter` (a concentrated core
  cut from a diffuse rest), :py:class:`nessai.partition.FirstOf` (the first
  of several), or any object with the same ``propose`` method.
- ``n_min`` (``nlive // 5``), ``max_regions`` (4), ``check_every``
  (``nlive // 20`` of each region), ``plot_every`` (``nlive``), ``plot``
  (True).

The partition can be switched on for a resumed run
(``FlowSampler(..., resume=True, partition=...)``).

Outputs
=======

- ``diagnostics/partition.png``: histograms of the split coordinates
  coloured by region, or, before the first split, by the parts of the cut
  the splitter would make. Saved every ``plot_every`` iterations and at each
  split (``partition_split_<iteration>.png``).
- One log line per split and per stopped region, and a line per region in
  the periodic log.
- In the result file: ``partitions`` (one row per region made: iteration,
  threshold, parent, share, numbers of points, ``ln X``, the cut),
  ``regions``, ``nested_samples_region`` (the region of each nested sample,
  ``-1`` before the first split) and ``region_log_evidence``.
- Each region's flow is saved in ``proposal/region_<label>/``.

Insertion-index tests are made per region; the run's p-value is the
smallest.
