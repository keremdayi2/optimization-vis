import torch

# Quadratic objective (x-x_0)^T P (x-x_0)
# if passing multiple vectors, make sure the rows are the decision variables
# ie, x[i] is agent i's decision variable and has the same dimension as x_0
class QuadraticObjective:
    def __init__(self, P, x_0):
        self.P = P
        self.x_0 = x_0
    
    # note!!! can modify torch computation graphs
    def __call__(self, x):
        x_p = x-self.x_0
        intermediate = (x_p @ self.P) * x_p
        return intermediate.sum(dim=-1)
   
    def plot(self, ranges = [[0, 5], [0, 5]]):
        with torch.no_grad():
            pass

# allows different objectives for different agents
class MultiAgentObjective:
    def __init__(self, objectives):
        self.objectives = objectives
    
    # note!!! can modify torch computation graphs
    def __call__(self, x):
        losses = torch.zeros(x.shape[0])
        for i in range(x.shape[0]):
            losses[i] = self.objectives[i](x[i])
        return losses
    
    # plot the sum (or average) of the objectives
    def plot(self, ranges = [[0, 5], [0,5]]):
        with torch.no_grad():
            pass
        
# we need this because otherwise we cannot backward losses of each agent
# because torch.backward() works only for a scalar loss, we need to iterate over the 
# agents' objectives and call backward() for each agent
def multi_backward(objective):
    for i, item in enumerate(objective):
        item.backward(retain_graph=True)