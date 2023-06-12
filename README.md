# optimization-vis
Visualization of some optimization algorithms

## Modules
`constrained-opt-vis`: contains implementation of some optimization algorithms for constrained quadratic problems with linear constraints. For now, the barrier method and the primal-dual method are implemented

`push-pull`: contains an implementation of the push-pull distributed optimization algorithm on directed graphs in pytorch

## General Structure

The reason we use pytorch is to be able to use automatic differentiation. This makes implementing gradient based methods so much easier and more flexible. For now, because of visualization purposes, we will work in $\mathbb{R}^2$. However, everything should work in $\mathbb{R}^n$, but you probably will have a harder time visualizing loss functions in $\R^n$ with $n > 2$.

To implement objectives and constraints, we have the `objective` and `constraint` modules. 

**Objectives**: `objective.QuadraticObjective` takes in a matrix $P$ and a vector $x_0$ (both torch Tensors) and implements the objective function 

$$(x-x_0)^T P (x-x_0)$$

ie. 

```
q = objective.QuadraticObjective(P, x_0)
q(x) # equals (x-x_0)^T P (x-x_0)
```

**Constraints**: 

## Gradient Implementation
In multi agent optimization algorithms, agents can only evaluate their own gradients. One way to do this is compute the objective for each agent and then call `.backward()` on the loss for each agent, and then operate on each agent separately. To make this process easier, we have a `objective.MultiAgentObjective` which allows us to evaluate the losses of multiple agents at the same time. (basically runs the above process). Hence, if `multi_loss = objective(x)`, it would make sense that each row of `x.grad` contains the gradients of the agent that corresponds to that row. Thus, after the calls
```
    n_agents = ...
    d = ...
    x = torch.zeros(n_agents, d) # each row corresponds to an agent's decision variable
    objectives = [...] # list of objectives
    multi_obj = MultiAgentObjective(objectives)
    loss = multi_obj(x)
    multi_backward(obj)
```

`x.grad` will contain the gradients $\nabla f_i(x_i)$. ie. `x.grad[i]` is $\nabla f_i(x_i)$. Below is a sample implementation of gradient descent for each agent simultaneously