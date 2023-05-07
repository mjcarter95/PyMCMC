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
    K = 100
    dim = 5
    num_steps = 5
    step_size = 0.1

    target = Target(
        dim=dim,
        mean=np.array([-4, 2, 0, 2, 4]),
        cov=np.eye(5),
    )

    sample_proposal = multivariate_normal(mean=np.zeros(target.dim), cov=np.eye(target.dim))
    momentum_proposal = multivariate_normal(mean=np.zeros(target.dim), cov=np.eye(target.dim))
    integrator = LeapfrogIntegrator(target=target, step_size=step_size)

    forward_kernel = HMCProposal(
        dim=target.dim,
        target=target,
        num_steps=num_steps,
        momentum_proposal=momentum_proposal,
        integrator=integrator,
    )

    # forward_kernel = RandomWalkProposal(
    #     target=target,
    #     dim=target.dim,
    # )

    rw_mcmc = MCMCSampler(
        K=K,
        dim=target.dim,
        target=target,
        forward_kernel=forward_kernel,
        sample_proposal=sample_proposal,
    )

    rw_mcmc.sample()

    print(f"Mean estimate:\n{np.mean(rw_mcmc.x, axis=0)}")
    print(f"Variance estimate:\n{np.var(rw_mcmc.x, axis=0)}")


if __name__ == "__main__":
    main()
