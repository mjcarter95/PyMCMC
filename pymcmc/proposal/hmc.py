from typing import Tuple

from scipy.stats import multivariate_normal

import autograd.numpy as np

import random


class HMCProposal:
    """Hamiltonian Monte Carlo Proposal

    Propagate samples using a Hamiltonian Monte Carlo (HMC) proposal. HMC propagates
    samples using an approximate simulation of the Hamiltonian dynamics of a system.

    [1] https://mc-stan.org/docs/2_19/reference-manual/hamiltonian-monte-carlo.html

    Attributes:
        target: Target distribution of interest.
        step_size: Step size for the leapfrog integrator.
        num_steps: Number of leapfrog steps to take.
        momentum_proposal: Momentum proposal distribution.
        random_num_steps: Whether or not to use a random number of leapfrog steps.
    """

    def __init__(
        self,
        target,
        num_steps: int,
        momentum_proposal,
        integrator,
        random_num_steps: bool = False,
    ):
        self.target = target
        self.num_steps = num_steps
        self.momentum_proposal = momentum_proposal
        self.integrator = integrator
        self.random_num_steps = random_num_steps

        self.dist = multivariate_normal(np.zeros(self.target.dim), np.eye(self.target.dim))

    def rvs(self, x_cond):
        """
        Description:
            Propogate a set of samples using Hamiltonian Monte Carlo (HMC).

        Args:
            x_cond: Current particle positions.

        Returns:
            x_prime: Updated particle positions.
            v_prime: Updated particle velocities.
            T: Length of Leapfrog trajectories.
        """
        v_cond = self.momentum_proposal.rvs()
        grad_x = self.target.logpdfgrad(x_cond)

        # Determine the number of leapfrog steps to take
        if self.random_num_steps:
            num_steps = int(np.ceil(np.random.exponential(scale=self.num_steps)))
        else:
            num_steps = self.num_steps

        # Take leapfrog steps
        x_prime, v_prime, grad_x_prime = self.integrator.step(x_cond, v_cond, grad_x)

        for _ in range(1, num_steps):
            x_prime, v_prime, grad_x_prime = self.integrator.step(
                x_prime, v_prime, grad_x_prime
            )

        U = self.target.logpdf(x_prime) - (0.5 * np.dot(v_prime, v_prime))
        K = self.target.logpdf(x_cond) - (0.5 * np.dot(v_cond, v_cond))
        acceptance_ratio = np.exp(U - K)
        acceptance_probability = min(1., acceptance_ratio)
        
        if random.uniform(0, 1) < acceptance_probability:
            return x_prime

        return x_cond

    def logpdf(self, v):
        """
        Description:
            Calculate the log probability of the forward kernel.

        Args:
            v: Particle velocities.

        Returns:
            log_prob: Log probability of the forward kernel.
        """

        return self.dist.logpdf(v)
