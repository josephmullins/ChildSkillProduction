# ChildSkillProduction
Child skill production: Accounting for parental and market-based time and goods investments

## Estimation Routine - Simple Example

### The likelihood model

The simplest version of the problem consists of three blocks in the likelihood. First, a vector of measures $M_t$ allows for noisy observations of skills:

$$M_{tj} = \lambda_{j}\log(\theta_{t}) + \zeta_{tj},\ \zeta_{tj}\sim\mathcal{N}(0,\sigma^2_{\zeta,j}).$$

Let $\beta_1=\{\lambda,\sigma_{\zeta}\}$ be the vector of measurement parameters. For identification we must normalize $\lambda_{1}=1$. Second, $\theta_{t+1}$ is determined according to investments and current skills in the parametric outcome equation:

$$ \theta_{1} = g(I,\theta_0;\beta_2)\eta, \eta\sim\mathcal{N}(0,\sigma^2_{\eta}).$$

Finally, assume that $\log(I)$ and $\log(\theta_0)$ are joint from $K$ mixtures of a joint normal:

$$ f([\log(\theta_1),I]) = \sum_{k=1}^K \pi_{k}\phi([\log(I),\log(\theta_0);\mu_{k},\Sigma_{k}) $$

Let $\beta_3=(\Sigma_{k},\mu_{k})_{k=1}^K$. The full maximum likelihood estimator is:

$$ (\beta_1,\beta_2,\beta_3,\pi) = \arg\max\sum_{i}\log\left(\sum_{k}\int f(M|\theta,\beta_1)f(\theta_1|I,\theta_0,\beta_2)f(\theta_0,I;\beta_3,k)d\etad\theta_0dI\pi_k\right)$$

To avoid this three-dimensional integral we combine the E-M algorithm with simulation.
