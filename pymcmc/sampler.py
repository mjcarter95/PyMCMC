from typing import Tuple

import autograd.numpy as np

from time import time
from tqdm import tqdm


class MCMCSampler:
    def __init__(
        self,
        num_warmup: int,
        num_samples: int,
        target,
        forward_kernel,
        sample_proposal,
        verbose: bool = False,
        seed: int = 0,
    ):
        self.num_warmup = num_warmup  # Number of warmup iterations
        self.num_samples = num_samples  # Number of samples to draw
        self.K = num_warmup + num_samples  # Total number of iterations
        self.target = target  # Target distribution
        self.forward_kernel = forward_kernel  # Forward kernel distribution
        self.sample_proposal = sample_proposal  # Initial sample proposal distribution
        self.verbose = verbose  # Show stdout
        self.seed = seed  # Random seed

        self.x = np.zeros([self.K, self.target.dim])

    def sample(self):
        """
        Sample from the target distribution using a Random Walk
        Markov Chain Monte Carlo sampler.
        """

        print(f"Sampling for {self.K} iterations")
        start_time = time()

        # Draw initial samples from the sample proposal distribution
        self.x[0] = self.sample_proposal.rvs()

        # Main sampling loop
        for k in tqdm(range(1, self.K)):
            if self.verbose:
                print(f"iter {k}")

            # Propose a new sample
            self.x[k] = self.forward_kernel.rvs(self.x[k - 1])

        time_taken = time() - start_time
        print(f"Finished sampling in {time_taken:5f} seconds")
