import torch


'''
    Check the vectorization operations in calling objectives. Should be able to
    easily generate contour maps
'''

class Objective:
    def __init__(self):
        pass

    #
    def global_objective(self, x):
        self.__call__(x)


# Quadratic objective (x-x_0)^T P (x-x_0)
# if passing multiple vectors, make sure the rows are the decision variables
# ie, x[i] is agent i's decision variable and has the same dimension as x_0
class QuadraticObjective:
    def __init__(self, P, x_0):
        self.P = P
        self.x_0 = x_0
        self.dims = x_0.shape[0]
    
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
        assert(len(objectives) > 0, 'please give a non-empty set of objectives')
        self.objectives = objectives
        self.dims = objectives[0].dims
        for o in objectives:
            assert(o.dims == self.dims, 'please make sure all agents have the same decision variable dimension')
    
    # note!!! can modify torch computation graphs
    def __call__(self, x):
        losses = torch.zeros(x.shape[0])
        for i in range(x.shape[0]):
            losses[i] = self.objectives[i](x[i])
        return losses
    
    # the global objective is the average of the losses
    def global_obj(self, x):
        loss = torch.zeros(1)
        for i in range(len(self.objectives)):
            loss += self.objectives[i](x)
        return loss/len(self.objectives)    


    def optimum(self, constraints = None, eta=0.1, max_num_iterations=500, epsilon = 1e-4):
        x_star = torch.zeros_like(self.objectives[0].x_0, dtype=torch.float64, requires_grad = True)
        prev_iteration = None
        for iteration in range(max_num_iterations):
            x_star.requires_grad = True
            loss = self.global_obj(x_star)
            loss.backward()

            with torch.no_grad():
                x_star -= eta * x_star.grad
                x_star.grad.zero_()

                if constraints != None:
                    x_star = constraints.project(x_star)


                if prev_iteration != None and torch.norm(prev_iteration - x_star) < epsilon:
                    return x_star
                prev_iteration = x_star.detach().clone()

        raise RuntimeError('Optimum could not found! Either decrease step size, increase number of iterations, or increase epsilon')

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