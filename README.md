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

Finally, assume that $\log(I)$ and $\log(\Psi_0)$ are drawn from $K$ mixtures of a joint normal. It will be convenient to represent this as:

$$ f(I) = \sum_{k=1}^K \pi_{k}\phi(\log(I)|\mu_{k},\Sigma_{k}) $$

where $\phi$ indicates the multi-variate normal density, with:

$$ \log(\Psi_{0}) = \mu_{\Psi,k} + \log(I)^{\prime}B_{k} + \epsilon_{\Psi},\ \epsilon_{\Psi,k}\sim\mathcal{N}(0,\sigma^2_{\Psi,k}) $$

where $k$ is the mixture that $I$ has been drawn from. Let $\beta_3=(\Sigma_{k},\mu_{k},\mu_{\Psi,k},B_{k},\sigma^2_{\Psi,k})_{k=1}^{K}$. 

### Estimation
The full maximum likelihood estimator is:

$$ \hat{\beta} = \widehat{(\beta_1,\beta_2,\beta_3,\pi)} = \arg\max\sum_{n}\log\left(\sum_{k}\int f(M_{n}|\Psi,\beta_1)f(\Psi_1|I_{n},\Psi_{0},\beta_2)f(\Psi_0,I_{n}|\beta_3,k)d\Psi_{1} d\Psi_0\pi_k\right)$$

To avoid this integral we combine the E-M algorithm with simulation. Fixing the current estimate at $\beta^{i}$, we can draw (for each $k$ and each $n$) $R$ samples of $\Psi_{1}^{nkr},\Psi_{0}^{nkr}$ from the conditional distribution $f(\Psi_{1},\Psi_{0}|\beta^{i},I)$ by first drawing $\Psi_{0}^{nkr}$ from $f(\Psi_{0}|\beta^{i}_{3},I_{n})$, then calculating: 

$$\Psi_{1}^{nkr} = g(I_{n},\Psi_{0}^{nkr};\beta_{2}^{i})\eta^{nkr} $$

where $\eta^{nkr}$ is drawn with standard deviation $\sigma^{i}_{\eta}$. Given these draws, we calculate posterior weights:

$$ w_{n}^{kr} = \frac{f(M_{n}|\Psi^{nkr},\beta_{1}^{i})f(I_{n}|k,\beta^{i}_{3})\pi_{k}^{i}}{ \sum_{k'} \sum_{r'}f(M_{n}|\Psi^{nk'r'},\beta_{1}^{i})f(I_{n}|k',\beta^{i}_{3})\pi_{k'}^{i}} $$

This is the "E step" of the algorithm, where we use simulation to approximate the integral. In the "M step", we choose each parameter $(\beta_1,\beta_2,\beta_3,\pi)$ to maximize the weighted log-likelihood given by the posterior weights $w_{n}^{kr}$. These can each be done separately as follows:

$$ \beta_{1}^{i+1} = \arg\max\sum_{n}\sum_{k}\sum_{r}\log(f(M_{n}|\Psi^{kr},\beta_{1})w_{n}^{kr} $$

$$\beta_{2}^{i+1} = \arg\max\sum_{n}\sum_{k}\sum_{r}\log(f(\Psi_{1}^{kr}|\Psi_{0}^{kr},I_{n},\beta_{2})w_{n}^{kr} $$

$$\beta_{3}^{i+1} = \arg\max\sum_{n}\sum_{k}\sum_{r}\log(f(\Psi_{0}^{kr},I_{n}|k,\beta_{3})w_{n}^{kr} $$

$$\pi_{k}^{i+1} = \frac{\sum_{n}\sum_{r}w_{n}^{kr}}{\sum_{n}\sum_{k'}\sum_{r}w_{n}^{k'r}} $$

where the last expression is the solution to log-likelihood maximization problem. It is also worth noting that the maximization problems for $\beta_{3}^{i+1}$ and $\beta_{1}^{i+1}$ also have a closed-form solution.

As mentioned above, these expressions use simulation to approximate the true expectation of the log-likelihood given $\beta^{i}$. For example, the first log-likelihood maximization problem approximates:

$$ \beta_{1}^{i+1} = \arg\max\sum_{n}\sum_{k}\int\int\log(f(M_{n}|\Psi,\beta_{1}))f(\Psi,k|M_{n},I_{n},\beta^{i})d\Psi_{1}d\Psi_{0} $$

The E-M routine jumps in between these expectation and maximization steps until $\beta^{i}$ and $\beta^{i+1}$ have converged according to some pre-set tolerance.

### More Details on the M-Step

For measurement parameters, the following solutions to the maximization problem hold:

$$ \lambda^{i+1}_{2} = \frac{\sum_{n,k,r}\sum_{t=0,1}w_{n}^{kr}\log(\Psi_{n,t}^{kr})M_{n,t,2}}{\sum_{n,k,r}\sum_{t=0,1}w_{n}^{kr}\log(\Psi_{n,t}^{kr})^{2}} $$

$$(\sigma_{\zeta,j}^{2})^{i+1} = \frac{\sum_{n,k,r}w_{n}^{kr}\sum_{t=0,1}(M_{n,t,j}-\lambda^{i+1}_{j}\log(\Psi_{n,t}^{kr}))^2}{\sum_{n,k,r}2w_{n}^{kr}} $$

For the initial distribution, the solution for $\pi^{i+1}$ is given explicitly above. The solution for the other parameters is as follows. The solution for the mean parameter for mixture $k$, $\mu_{k}$, is a weighted mean:

$$\mu_{k}^{i+1} = \frac{\sum_{n,r}w_{n}^{kr}\tilde{I}_{n}}{\sum_{n,r}w_{n}^{kr}} $$

where $\tilde{I}_{n} = \log(I_{n})$. The solution for the $(l,m)^{th}$ component of the covariance matrix $\Sigma_{k}$ is a sample covariance:


$$ \Sigma_{k}^{i+1}(l,m) = \frac{\sum_{n,r}w_{n}^{kr}(\tilde{I}_{n}(l)-\mu_{k}^{i+1}(l))(\tilde{I}_{n}(m)-\mu_{k}^{i+1}(m))}{\sum_{n,r}w_{n}^{kr}} $$

where $\tilde{I}_{n}(l)$ denotes the $l^{th}$ component of the vector $\tilde{I}_{n}$ and so on. Let $C_{k} = [\mu_{\Psi,k},B_{k}^\prime]^\prime$. The $M$-step solution for $C_{k}$ is a weighted regression:

$$C^{i+1}_{k} = \left(\sum_{n,r}w_{n}^{kr}\mathbf{x}_{n}\mathbf{x}_{n}^\prime\right)^{-1}\sum_{n,r}w_{n}^{kr}\mathbf{x}_{n}\log(\Psi_{n,0}^{kr}) $$

where $\mathbf{x}_{n} = [1,\ \tilde{I}_{n}^\prime]^\prime$ is the vector of log investments with a constant added. Finally, the $M$-step solution for $\sigma^2_{\Psi}$ is a weighted mean of squared residuals:

$$(\sigma^2_{\Psi,k})^{i+1} = \frac{\sum_{n,r}w_{n}^{kr}(\log(\Psi_{n,0}^{kr})-\mathbf{x}_{n}^{\prime}C^{i+1}_{k})^2}{\sum_{n,r}w_{n}^{kr}}.$$

## Diagnosing Convergence Issues
In theory, each step of the E-M routine is supposed to move up the likelihood monotonically. Although we cannot evaluate the exact log-likelihood, we can check the simulated log-likelihood that we are, in principle, maximizing. The approximate likelihood for each individual can be evaluated as:

$$\hat{f}^{R}(M_{n},I_{n}|\Theta) = \sum_{k}\sum_{r}f(M_{n}|\Psi^{nkr},\beta_{1})f(I_{n}|k,\beta)\pi_{k}$$

where each $(\Psi_{1}^{nkr},\Psi_{0}^{nkr})$ is one of $R$ simulation draws from $f(\Psi_{1},\Psi_{0}|\beta,I_{n})$ performed in the exact same way as it is done for the $E$ step. This approximates the true likelihood:

$${f}(M_{n},I_{n}|\Theta) = \sum_{k}\int f(M_{n}|\Psi,\beta_{1})f(\Psi_{1},\Psi_{0}|\beta,I_{n})f(I_{n}|k,\beta)\pi_{k}d\Psi_{0}d\Psi_{1}.$$

Thus, to check convergence at each step $i$, calculate:

$$ L^{R} = \sum_{n}\log\left(\hat{f}^{R}(M_{n},I_{n}|\Theta^{i})\right) $$

after each EM step, using the same draws of $\Psi_{1}$ and $\Psi_{0}$ that are used to calculate the posterior weights $w_{n}^{kr}$ at the E step.

# Alternative estimator

An alternative to full maximum likelihood is to first estimate the distribution of initial conditions and measurement parameters, and then use it to construct moments for a second stage estimate of production paremeters. Something to consider?