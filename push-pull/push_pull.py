import torch

import objective
import constraint

# compute the gradient error
#
# S(y, pi) = \sqrt{\sum_{i=1}^n \pi_i \|\frac{y_i}{\pi_i}-\sum_{l=1}^n y_l\|^2}
# 
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

