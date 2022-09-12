# ChildSkillProduction
Child skill production: Accounting for parental and market-based time and goods investments

## Estimation Routine - Simple Example

### The likelihood model

The simplest version of the problem consists of three blocks in the likelihood. First, a vector of measures $M_t$ allows for noisy observations of skills:

$$M_{tj} = \lambda_{j}\log(\theta_{t}) + \zeta_{tj},\ \zeta_{tj}\sim\mathcal{N}(0,\sigma^2_{\zeta,j}).$$

Let $\beta_1=\{\lambda,\sigma_{\zeta}\}$ be the vector of measurement parameters. For identification we must normalize $\lambda_{1}=1$. Second, $\theta_{t+1}$ is determined according to investments and current skills in the parametric outcome equation:

$$ \theta_{1} = g(I,\theta_0;\beta_2)\eta,\ \eta\sim\mathcal{N}(0,\sigma^2_{\eta}).$$

Finally, assume that $\log(I)$ and $\log(\theta_0)$ are joint from $K$ mixtures of a joint normal:

$$ f([\log(\theta_1),I]) = \sum_{k=1}^K \pi_{k}\phi([\log(I),\log(\theta_0);\mu_{k},\Sigma_{k}) $$

Let $\beta_3=(\Sigma_{k},\mu_{k})_{k=1}^K$. The full maximum likelihood estimator is:

$$ \hat{\beta} = \widehat{(\beta_1,\beta_2,\beta_3,\pi)} = \arg\max\sum_{n}\log\left(\sum_{k}\int f(M_{n}|\theta,\beta_1)f(\theta_1|I_{n},\theta_{0},\beta_2)f(\theta_0,I_{n};\beta_3,k)d\eta d\theta_0\pi_k\right)$$

To avoid this integral we combine the E-M algorithm with simulation. Fixing the current estimate at $\beta^{i}$, we can draw (for each $k$) $R$ samples of $\eta^{kr},\theta_{0}^{kr}$ from the distribution implied by $\beta_{3}^{i}$, and calculate posterior weights:

$$ w_{n}^{kr} = \frac{f(M_{n}|\theta^{kr},\beta_{1}^{i})f(\theta_{1}^{kr}|I_{n},\theta_{0}^{kr},\beta^{i}_2)f(\theta^{kr}_{0},I_{n};\beta^{i}_3,k)\pi_k}{ \sum__{k'} \sum_{r'}f(M_{n}|\theta^{k'r'},\beta_{1}^{i})f(\theta_{1}^{k'r'}|I_{n},\theta_{0}^{k'r'},\beta^{i}_2)f(\theta^{k'r'}_{0},I_{n};\beta^{i}_3,k')\pi_{k'}} $$

This is the "E step" of the algorithm. In the "M step", we choose each parameter $(\beta_1,\beta_2,\beta_3,\pi)$ to maximize the weighted log-likelihood given by the posterior weights $w_{n}^{kr}$. These can each be done separately as:

