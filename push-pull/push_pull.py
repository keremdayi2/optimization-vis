import sys
sys.path.insert(0, '..')

import torch
import objective
import constraint 


'''
	TODO: Maybe move the helpers into a helper class?
    1) create a method for push pull step, so that we can allow R and C to change. 
        It will give more control to the user
'''

# compute the gradient error
#
# S(y, pi) = \sqrt{\sum_{i=1}^n \pi_i \|\frac{y_i}{\pi_i}-\sum_{l=1}^n y_l\|^2}
# 

class Metrics:
    def grad_error(y, pi):
        with torch.no_grad():
            total_y = y.sum(dim=0)
            scaled_y = y/pi[:, None]
            diff = scaled_y - total_y
            errors = (diff * diff).sum(dim=1)
            return torch.sqrt(torch.dot(errors, pi))

    # compute the consensus error
    def consensus_error(x, phi):
        with torch.no_grad():
            x_hat = (x * phi[:, None]).sum(dim=0)
            diff = x - x_hat
            errors = (diff * diff).sum(dim=1)
            return torch.sqrt(torch.dot(errors, phi))

    def optimality_error(x, x_star, phi):
        with torch.no_grad():
            x_hat = (x * phi[:, None]).sum(dim=0)
            diff = x_hat - x_star
            return torch.sqrt(torch.dot(diff, diff))

class Helpers:
    # returns tuple (R, C)
    def construct_RC(A):
        # row and column stochastic matrices
        A.fill_diagonal_(1)
        A = A.T # take the transpose since this is the direction of information flow

        R = A/A.sum(axis=1)[:, None]
        C = A/A.sum(axis=0)
        return (R, C)

    # returns tuple (phi, pi)
    def find_eigenvectors(R, C):
        # find the left and right eigenvectors with eigenvalue 1 of R and C respectively.
        phi = torch.ones(1, R.shape[0], dtype=torch.float64)
        pi = torch.ones(C.shape[0], 1, dtype=torch.float64)

        for i in range(100):
            phi = phi @ R
            pi = C @ pi

        # make sure these are actually eigenvectors
        assert(torch.all(torch.isclose(pi, C @ pi)))
        assert(torch.all(torch.isclose(phi, phi @ R)))

        phi, pi = phi.flatten(), pi.flatten()
        phi = phi/phi.sum()
        pi = pi/pi.sum()

        return (phi, pi)

    '''
        Maybe change this into a push_pull_step() function so that
        it is easier to customize. For instance, if R, C changes over time.

        Could also easily implement robust push pull
    '''


def simulate_push_pull(R, C, obj, x_init = None, cnstr = None, eta = 0.01, num_iterations = 20):
    phi, pi = Helpers.find_eigenvectors(R, C)
    dimension = obj.objectives[0].x_0.shape[0] 
    
    if x_init == None:
        x_init = torch.zeros(R.shape[0], dimension)
    x = x_init
    
    loss = obj(x)
    objective.multi_backward(loss)
    y = x.grad.detach().clone()
    prev_grad = y.clone()
    x.grad.zero_()
    
    x_star = obj.optimum(cnstr)
    
    consensus_errors = []
    gradient_errors = []
    optimality_errors = []
    xs = []
    ys = []
    
    for iteration in range(num_iterations):
        # update x (disable gradients to not break things)
        consensus_errors.append(Metrics.consensus_error(x, phi))
        gradient_errors.append(Metrics.grad_error(y, pi))
        optimality_errors.append(Metrics.optimality_error(x, x_star, phi))
        ys.append(y.detach().clone())
        xs.append(x.detach().clone())

        with torch.no_grad():
            x = R @ x - eta * y
            if cnstr != None:
                x = cnstr.project(x)

        # update the new gradient
        x.requires_grad = True
        loss = obj(x)
        objective.multi_backward(loss)
        new_grad = x.grad.detach().clone() 
        x.grad.zero_()

        # update y
        y = (C @ y) + new_grad - prev_grad
        prev_grad = new_grad
    
    data = {}
    data['consensus_error'] = torch.Tensor(consensus_errors)
    data['grad_error'] = torch.Tensor(gradient_errors)
    data['optimality_gap'] = torch.Tensor(optimality_errors)
    data['xs'] = xs
    data['ys'] = ys
    return data