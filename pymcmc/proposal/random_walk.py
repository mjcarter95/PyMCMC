import autograd.numpy as np

from scipy.stats import multivariate_normal


class RandomWalkProposal:
    """Random Walk Proposal

    Propagate samples using a Gaussian random walk proposal.

    Attributes:
        dim: Dimensionality of the system.
        target: Target distribution of interest.
    """

    def __init__(
        self, target, dim: int, cov: float = 1.0,
    ):
        self.dim = dim  # Dimensionality of the target
        self.cov = cov  # Covariance of the Gaussian noise
        self.target = target  # Target distribution

        cov_matrix = np.multiply(np.eye(self.dim), self.cov)
        self.dist = multivariate_normal(np.zeros(self.dim), cov_matrix)

    def rvs(self, x_cond):
        """
        Description:
            Propogate a set of samples using Gaussian random walk.

        Args:
            x_cond: Current particle positions.

        Returns:
            x_prime: Updated particle positions.
        """

        # Propose a new sample
        x_proposed = x_cond + self.dist.rvs()

        # Calculate the acceptance ratio
        log_acceptance_ratio = (
            self.target.logpdf(x_proposed)
            - self.target.logpdf(x_cond)
            + self.logpdf(x_cond, x_proposed)
            - self.logpdf(x_proposed, x_cond)
        )

        # Accept or reject the proposed sample
        if np.log(np.random.uniform()) < log_acceptance_ratio:
            return x_proposed

        return x_cond 

    def logpdf(self, x, x_new):
        """
        Description:
            Calculate the log probability of the forward kernel.

        Args:
            x: Current particle positions.
            x_new: New particle positions.

        Returns:
            log_prob: Log probability of the forward kernel.
        """

        return self.dist.logpdf(x_new - x)
