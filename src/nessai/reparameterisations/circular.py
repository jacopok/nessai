# -*- coding: utf-8 -*-
"""
Reparameterisation for periodic parameters sampled with a circular flow.
"""

import logging

import numpy as np

from .base import Reparameterisation

logger = logging.getLogger(__name__)

TWO_PI = 2.0 * np.pi


def wrap_angle(x):
    """Wrap to ``[-pi, pi)``; ``np.mod`` can round up to the upper end."""
    y = np.mod(x + np.pi, TWO_PI) - np.pi
    return np.where(y >= np.pi, -np.pi, y)


class Circular(Reparameterisation):
    """Periodic parameters as angles on ``[-pi, pi)``.

    Each parameter is mapped linearly from its prior range, which is taken to
    be one period, onto ``[-pi, pi)``. The outputs are listed in
    :py:attr:`circular_parameters`, which ``FlowProposal`` passes to the flow
    as its circular features: the flow must be a circular flow
    (``ftype='circular'``, see
    :py:class:`~nessai.flows.circular.CircularNeuralSplineFlow`). Unlike
    :py:class:`~nessai.reparameterisations.Angle`, no auxiliary radius is
    added and the flow's density is continuous across the period boundary.

    Parameters
    ----------
    parameters : str or list
        Periodic parameters.
    prior_bounds : dict
        Prior bounds; the range of each is its period.
    """

    requires_bounded_prior = True

    def __init__(
        self,
        input_parameters=None,
        output_parameters=None,
        persistent_parameters=None,
        auxiliary_parameters=None,
        prior_bounds=None,
        rng=None,
        inverse_input_parameters=None,
        parameters=None,
    ):
        super().__init__(
            input_parameters=input_parameters,
            output_parameters=output_parameters,
            persistent_parameters=persistent_parameters,
            auxiliary_parameters=auxiliary_parameters,
            prior_bounds=prior_bounds,
            rng=rng,
            inverse_input_parameters=inverse_input_parameters,
            parameters=parameters,
        )
        if output_parameters is None:
            self.output_parameters = [f"{p}_circ" for p in self.parameters]
        self._low = {p: self.prior_bounds[p][0] for p in self.parameters}
        self._scale = {
            p: TWO_PI / np.ptp(self.prior_bounds[p]) for p in self.parameters
        }

    @property
    def circular_parameters(self):
        """Output parameters that are angles on ``[-pi, pi)``."""
        return list(self.output_parameters)

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        """Map the parameters onto ``[-pi, pi)``."""
        for p, pp in zip(self.parameters, self.output_parameters):
            value = self.get_parameter_value(p, x, x_prime)
            x_prime[pp] = wrap_angle(
                (value - self._low[p]) * self._scale[p] - np.pi
            )
            log_j += np.log(self._scale[p])
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        """Map angles on the real line back onto the prior range."""
        for p, pp in zip(self.parameters, self.output_parameters):
            value = (wrap_angle(x_prime[pp]) + np.pi) / self._scale[
                p
            ] + self._low[p]
            x, x_prime = self.set_parameter_value(p, value, x, x_prime)
            log_j -= np.log(self._scale[p])
        return x, x_prime, log_j
