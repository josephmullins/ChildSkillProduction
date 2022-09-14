# ChildSkillProduction
This code excutes the joint estimation routines in Caucutt, Lochner, Mullins & Park (current draft 2020). The goal is to estimate a production function with a nested CES structure. For married couples it takes the general form:

$$ \Psi_{t+1} = \theta\left[\left(a_{m}\tau_{m,t}^{\rho} + a_{f}\tau_{f,t}^{\rho} + a_{g}g_{t}^{\rho}\right)^{\gamma/\rho} + a_{Y_c}Y_{c,t}^{\gamma}\right]^{\delta_{1}/\gamma}\Psi_{t}^{\delta_2} $$

where

 - $\tau_{m}$ and $\tau_{f}$ are the mother's and father's time inputs,
 - $g$ are goods inputs inside the home, and
 - $Y_{c,t}$ is a childcare input outside the home.

## Estimation Routine - Simple Example

This section describes the estimation routine for a simplified version of the model.

### The likelihood model

The simplest version of the problem consists of three blocks in the likelihood. First, a vector of measures $M_t$ allows for noisy observations of skills:

$$M_{tj} = \lambda_{j}\log(\Psi_{t}) + \zeta_{tj},\ \zeta_{tj}\sim\mathcal{N}(0,\sigma^2_{\zeta,j}).$$

Let $\beta_1=\{\lambda,\sigma_{\zeta}\}$ be the vector of measurement parameters. For identification we must normalize $\lambda_{1}=1$. Second, $\theta_{t+1}$ is determined according to investments and current skills in the parametric outcome equation:

$$ \Psi_{1} = g(I,\Psi_{0};\beta_2)\eta,\ \log(\eta)\sim\mathcal{N}(0,\sigma^2_{\eta}).$$

Finally, assume that $\log(I)$ and $\log(\Psi_0)$ are drawn from $K$ mixtures of a joint normal:

$$ f(\Psi_1,I) = \sum_{k=1}^K \pi_{k}\phi([\log(I),\log(\Psi_0)]|\mu_{k},\Sigma_{k}) $$

Let $\beta_3=(\Sigma_{k},\mu_{k})_{k=1}^{K}$. 

### Estimation

The full maximum likelihood estimator is:

$$ \hat{\beta} = \widehat{(\beta_1,\beta_2,\beta_3,\pi)} = \arg\max\sum_{n}\log\left(\sum_{k}\int f(M_{n}|\Psi,\beta_1)f(\Psi_1|I_{n},\Psi_{0},\beta_2)f(\Psi_0,I_{n}|\beta_3,k)d\Psi_{1} d\Psi_0\pi_k\right)$$

To avoid this integral we combine the E-M algorithm with simulation. Fixing the current estimate at $\beta^{i}$, we can draw (for each $k$) $R$ samples of $\Psi_{1}^{kr},\Psi_{0}^{kr}$ from the distribution implied by $\beta^{i}$, and calculate posterior weights:

$$ w_{n}^{kr} = \frac{f(M_{n}|\Psi^{kr},\beta_{1}^{i})\pi_{k}^{i}}{ \sum_{k'} \sum_{r'}f(M_{n}|\Psi^{k'r'},\beta_{1}^{i})\pi_{k'}^{i}} $$

This is the "E step" of the algorithm, where we use simulation to approximate the integral. In the "M step", we choose each parameter $(\beta_1,\beta_2,\beta_3,\pi)$ to maximize the weighted log-likelihood given by the posterior weights $w_{n}^{kr}$. These can each be done separately as follows:

$$ \beta_{1}^{i+1} = \arg\max\sum_{n}\sum_{k}\sum_{r}\log(f(M_{n}|\Psi^{kr},\beta_{1})w_{n}^{kr} $$

$$\beta_{2}^{i+1} = \arg\max\sum_{n}\sum_{k}\sum_{r}\log(f(\Psi_{1}^{kr}|\Psi_{0}^{kr},I_{n},\beta_{2})w_{n}^{kr} $$

$$\beta_{3}^{i+1} = \arg\max\sum_{n}\sum_{k}\sum_{r}\log(f(\Psi_{0}^{kr},I_{n}|k,\beta_{3})w_{n}^{kr} $$

$$\pi_{k}^{i+1} = \frac{\sum_{n}\sum_{r}w_{n}^{kr}}{\sum_{n}\sum_{k'}\sum_{r}w_{n}^{k'r}} $$

where the last expression is the solution to log-likelihood maximization problem. It is also worth noting that the maximization problems for $\beta_{3}^{i+1}$ and $\beta_{1}^{i+1}$ also have a closed-form solution.

As mentioned above, these expressions use simulation to approximate the true expectation of the log-likelihood given $\beta^{i}$. For example, the first log-likelihood maximization problem approximates:

$$ \beta_{1}^{i+1} = \arg\max\sum_{n}\sum_{k}\int\int\log(f(M_{n}|\Psi,\beta_{1}))f(\Psi,k|M_{n},I_{n},\beta^{i})d\Psi_{1}d\Psi_{0} $$

The E-M routine jumps in between these expectation and maximization steps until $\beta^{i}$ and $\beta^{i+1}$ have converged according to some pre-set tolerance.

Hello! This is a test to see if push requests are functioning properly. -Maddi

