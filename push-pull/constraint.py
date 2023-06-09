import torch

'''
    TODO: specify what properties each constraint class should have. Also establish some inheritance structure.
    For instance, constraint objects should probably have a feasible(self, x) method so that they can be plotted
'''

# class RectangleConstraint
# ranges is a list in the form
# [(l1, u1), (l2,u2), ...]
# where (ln, un) are the lower and upper bounds for dimension n
class RectangleConstraint():
    def __init__(self, ranges):
        self.lowers = torch.tensor([r[0] for r in ranges])
        self.uppers = torch.tensor([r[1] for r in ranges])
    
    # return the projection of x onto the convex set described by the constraints above
    # projection works when you supply a matrix where row i is agent i's decision variable vector
    def project(self, x, with_grad = False):
        if not with_grad:
            with torch.no_grad():
                x = torch.minimum(x, self.uppers)
                x = torch.maximum(x, self.lowers)
                return x
        else:
            x = torch.minimum(x, self.uppers)
            x = torch.maximum(x, self.lowers)
            return x

    # return true if point is feasible
    # again, each row of x can be an agent's vector and will return a vector whose element i is true 
    # iff agent x's vector is feasible
    def feasible(self, x):
        with torch.no_grad():
            constraint_sat = torch.cat([x >= self.lowers, x <= self.uppers], dim=-1)
            return torch.all(constraint_sat, dim=1)

class SphericalConstraint():
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
    
    def project(self, x):
        with torch.no_grad():
            unsqueezed = len(x.shape) == 1
            if unsqueezed:
                x = torch.unsqueeze(x, 0)
            diff = x - self.center
            norms = torch.norm(diff, dim=1)
            for i in range(diff.shape[0]):
                if norms[i] > self.radius:
                    diff[i] /= norms[i] # normalize the radius to 1
                    diff[i] *= self.radius
            
            if unsqueezed:
                return (self.center + diff)[0, :]
            return self.center + diff
    
    def feasible(self, x):
        with torch.no_grad():
            diff = x - self.center
            return torch.norm(diff, dim = 1) <= self.radius