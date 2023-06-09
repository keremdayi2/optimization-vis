import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import torch

'''
	TODO:
	1) objective contour plot
	2) feasible set plot
	3) animation for iterative optimization
	4) plot gradients/gradient tracking variables
'''

def plot_problem_2d(objective, constraints=None, rng=[[0, 7], [0, 5]], cell_size = 0.1):
    with torch.no_grad():
        dims = objective.dims
        assert dims == 2, 'plot_problem_2d is only intended for 2d visualizations'
        x_low, x_high = rng[0]
        y_low, y_high = rng[1]
        
        assert x_high > x_low and y_high > y_low, 'please enter a proper range for plotting'

        # construct the grid
        nx = int((x_high - x_low)/cell_size) + 1
        ny = int((y_high - y_low) / cell_size) + 1
        
        xs = torch.linspace(x_low, x_high, nx)
        ys = torch.linspace(y_low, y_high, ny)

        coords = torch.meshgrid([xs, ys])
        coords = torch.stack([coords[0], coords[1]])

        coords = torch.permute(coords, (2, 1, 0)) # permute to (y, x, vec)
        f = torch.zeros(coords.shape[0], coords.shape[1])

        feasibilities = torch.ones(coords.shape[0], coords.shape[1])

        # plot the objective and constraints
        for i in range(ny):
            for j in range(nx):
                f[i][j] = objective.global_obj(coords[i][j])
                if constraints != None:
                    coord = coords[i][j].unsqueeze(0)
                    feasibilities[i][j] *= constraints.feasible(coord)[0]

        plt.contour(coords[:, :, 0], coords[:, :, 1], f)

        feasibilities *= 50
        # plot the feasible region
        if constraints != None:
            plt.imshow(feasibilities, extent=(x_low, x_high, y_low, y_high),vmin=0, vmax =100, origin="lower", cmap="Blues",)
    