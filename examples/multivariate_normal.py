import autograd.numpy as np

from scipy.stats import multivariate_normal
from autograd import elementwise_grad as egrad
from autograd.scipy import stats as AutoStats

from pymcmc.sampler import MCMCSampler
from pymcmc.proposal.random_walk import RandomWalkProposal

from pymcmc.proposal.hmc import HMCProposal
from pymcmc.integrator.leapfrog import LeapfrogIntegrator


class Target():
    def __init__(self, dim, mean, cov):
        self.mean = mean
        self.cov = cov
        self.dim = dim

    def logpdf(self, x):
        return AutoStats.multivariate_normal.logpdf(x, mean=self.mean, cov=self.cov)

    def logpdfgrad(self, x):
        grad = egrad(self.logpdf)
        return grad(x)


def main():
    n_chains = 4
    num_warmup = 500
    num_samples = 500
    dim = 5

    target = Target(
        dim=dim,
        mean=np.array([-4, 2, 0, 2, 4]),
        cov=np.eye(5),
    )

    sample_proposal = multivariate_normal(mean=np.zeros(target.dim), cov=np.eye(target.dim))

    # Initialise the RW proposal
    forward_kernel = RandomWalkProposal(
        target=target,
    )

    # Run the RW sampler for each chain
    rw_mcmc_chains = []
    for i in range(n_chains):
        rw_mcmc_chains.append(
            MCMCSampler(
                num_warmup=num_warmup,
                num_samples=num_samples,
                target=target,
                forward_kernel=forward_kernel,
                sample_proposal=sample_proposal,
            )
        )

        rw_mcmc_chains[i].sample()

    # Filter out the warmup samples, concatenate the chains
    rw_mcmc_samples = np.concatenate([chain.x[num_warmup:] for chain in rw_mcmc_chains], axis=0)

    print(f"RW Mean estimate:\n{np.mean(rw_mcmc_samples, axis=0)}")
    print(f"RW Variance estimate:\n{np.var(rw_mcmc_samples, axis=0)}")

    # Initialise the HMC proposal
    num_steps = 5
    step_size = 0.1
    momentum_proposal = multivariate_normal(mean=np.zeros(target.dim), cov=np.eye(target.dim))
    integrator = LeapfrogIntegrator(target=target, step_size=step_size)
    forward_kernel = HMCProposal(
        target=target,
        num_steps=num_steps,
        momentum_proposal=momentum_proposal,
        integrator=integrator,
    )

    # Run the HMC sampler for each chain
    hmc_mcmc_chains = []
    for i in range(n_chains):
        hmc_mcmc_chains.append(
            MCMCSampler(
                num_warmup=num_warmup,
                num_samples=num_samples,
                target=target,
                forward_kernel=forward_kernel,
                sample_proposal=sample_proposal,
            )
        )

        hmc_mcmc_chains[i].sample()

    # Filter out the warmup samples, concatenate the chains
    hmc_mcmc_samples = np.concatenate([chain.x[num_warmup:] for chain in hmc_mcmc_chains], axis=0)

    print(f"HMC Mean estimate:\n{np.mean(hmc_mcmc_samples, axis=0)}")
    print(f"HMC Variance estimate:\n{np.var(hmc_mcmc_samples, axis=0)}")


if __name__ == "__main__":
    main()
