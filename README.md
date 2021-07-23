# R-SVRC

 This file contains the Riemannian Stochastic Variance-Reduced Cubic-Regularized Newton Method aiming to solve the following problem
$$
\min_{\mathbf{x}\in\mathcal{M}\subseteq\mathbb{R}^{m\times n}} f(\mathbf{x}).
$$
For more information, please refer to   [Zhang, D., Davanloo Tajbakhsh, S. (2020), Riemannian Stochastic Variance-Reduced Cubic-Regularized Newton Method](https://arxiv.org/abs/2010.03785).

## Content

```
.
├── manopt // Matlab toolbox 
├── plots // results of current numerical examples 
├── Experiments 
│   ├── copula_SDP // Parameter estimation of Student t distribution 
│   │   ├── copula_SDP_methodsComparison.m // Compare with different Riemannian second-order methods
│   │   ├── copula_SDP_methodsSimulation.m // Performances under different parameter setting 
│   │   ├── copula_SDP_methodsOther.m // Performances with different noise-level data 
│   └── sphere // Regression over Sphere manifold
│       ├── sphere_methodsComparison.m
│       ├── sphere_SDP_methodsSimulation.m
│       ├── sphere_SDP_methodsOther.m
├── rsvrc.m //Solver for the Riemannian stochastic variance-reduced cubic-regularized Newton method
└── others 
    ├── sphere_3DPlotting.m // 
    └── plot_distribution_prctile.m // 


```

## Roadmap 

Please add the path of MANOPT first to enable the basic operations for Riemannian optimization. Then, you can click, for example, copula_SDP_methodsComparison.m, to see the comparision of different methods for the parameter estimation problem of Student t distribution, w.r.t. cpu time cost and number of second-order oracles required. 

## Call the function

Function:

```
function [x, cost, info, options]=rsvrc(problem,x,options);
```

solves problem
$$
\min_{\mathbf{x}\in\mathcal{M}\subseteq\mathbb{R}^{m\times n}} f(\mathbf{x})
$$
Inputs:

- problem: structure of problem. 
- x: initial point on the manifold (optional).
- options: the options structure is used to overwrite the default values. All options have a default value and are hence optional. 
  - maxepoch (20): the algorithm terminates if maxepoch epochs have been executed. 
  - maxinneriter(5): the maximum length of each epoch. 
  - tolgradnorm(1e-6): the algorithm terminates if the norm of the gradient drops below this. 
  - stochastic(1): run the stochastic method if 1; otherwise, run the plain cubic regularizied Newton method. 
  - batchsize_g (N/20): batchsize of $b_g$.
  - batchsize_h (N/20): batchsize of $b_h$. 
  - subproblemsolver(@arc_conjugate_gradient): function handle to solve the cubic subproblem. 
  - sigma_0: initial size of the cubic regularized parameter. 


Outputs:

- x: solution of the algorithm.
- cost: objective function cost associated with the solution x.
- info: the struct-array info contains information about the iterations
  - iter: number of iterations 
  - cost: objective function cost associated with current iterate 
  - gradnorm: norm of gradient of objective function at current iterate
  - so_count: number of second-order oracle calling 
  - time_cpu: cpu time used by now 
- options: report the options of the parameters.  
