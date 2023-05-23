# PyMCMC
### A collection of MCMC methods in Python using Numpy and Scipy

<!-- tempate https://github.com/scottydocs/README-template.md/blob/master/README.md -->
![GitHub repo size](https://img.shields.io/github/repo-size/mjcarter95/PyMCMC)
![GitHub contributors](https://img.shields.io/github/contributors/mjcarter95/PyMCMC)
![GitHub stars](https://img.shields.io/github/stars/mjcarter95/PyMCMC?style=social)
![GitHub forks](https://img.shields.io/github/forks/mjcarter95/PyMCMC?style=social)
![Twitter Follow](https://img.shields.io/twitter/follow/mjcarter955?style=social)

PyMCMC allows users to sample from probability distributions of interest using various Markov Chain Monte Carlo methods.
<!-- 
## Prerequisites

Before you begin, ensure you have met the following requirements:-->
<!--- These are just example requirements. Add, duplicate or remove as required --->
<!-- * You have installed the latest version of `<coding_language/dependency/requirement_1>`
* You have a `<Windows/Linux/Mac>` machine. State which OS is supported/which is not.
* You have read `<guide/link/documentation_related_to_project>`. -->

## Installing PyMCMC

To install PyMCMC, follow these steps:

```
pip install pip@git+https://github.com/mjcarter95/PyMCMC.git
```

## Using PyMCMC

A number of example problems are provided in the `examples` folder.

Example: A single Metropolis-Hastings chain can be applied to a user-defined target density as follows

```
    target = Target()

    sample_proposal = multivariate_normal(mean=np.zeros(target.dim), cov=np.eye(target.dim))
    momentum_proposal = multivariate_normal(mean=np.zeros(target.dim), cov=np.eye(target.dim))

    rw_mcmc = MCMCSampler(
        K=K,
        dim=target.dim,
        target=target,
        forward_kernel=forward_kernel,
        sample_proposal=sample_proposal,
    )

    rw_mcmc.sample()
```

## Contributing to PyMCMC
<!--- If your README is long or you have some specific process or steps you want contributors to follow, consider creating a separate CONTRIBUTING.md file--->
To contribute to PyMCMC, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Contributors

Thanks to the following people who have contributed to this project:

* [@mjcarter95](https://github.com/mjcarter95) 📖
<!-- * [@vberaud](https://github.com/vberaud) 🐛  -->

<!-- 
You might want to consider using something like the [All Contributors](https://github.com/all-contributors/all-contributors) specification and its [emoji key](https://allcontributors.org/docs/en/emoji-key). -->

## Contact

If you want to contact me you can reach me at <m (dot) j (dot) carter (at) liverpool (dot) ac (dot) uk>.

## Citation
We appreciate citations as they let us discover what people have been doing with the software. 

To cite PyMCMC in publications use:

Carter, M. (2023). PyMCMC (1.0.0). https://pypi.org/project/pysmc

Or use the following BibTeX entry:

```
@misc{pymcmc,
  title = {PyMCMC (1.0.0)},
  author = {Carter, Matthew},
  year = {2023},
  month = may,
  howpublished = {PyPI}
}
```

## License
<!--- If you're not sure which open license to use see https://choosealicense.com/--->

This project uses the following license: [<license_name>](<link>).
