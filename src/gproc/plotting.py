import numpy as np
import matplotlib.pyplot as plt

def density_contourf(ax, density, x1_lim=[-1, 1], x2_lim=[-1, 1], granularity=30):
    xs = np.linspace(x1_lim[0], x1_lim[1], granularity)
    ys = np.linspace(x2_lim[0], x2_lim[1], granularity)
    x, y = np.meshgrid(xs, ys, indexing='xy')
    points = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])

    z = density(points).reshape(granularity, granularity)
    
    return ax.contourf(x, y, z, cmap=plt.get_cmap('Oranges'))
    
def density_contour(ax, density, x1_lim=[-1, 1], x2_lim=[-1, 1], cmap=plt.get_cmap('Purples'), granularity=30):
    xs = np.linspace(x1_lim[0], x1_lim[1], granularity)
    ys = np.linspace(x2_lim[0], x2_lim[1], granularity)
    x, y = np.meshgrid(xs, ys, indexing='xy')
    points = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])

    z = density(points).reshape(granularity, granularity)
        
    return ax.contour(x, y, z, cmap=cmap)

def contour_2d(ax, sampler, N, x1_lim=[-1, 1], x2_lim=[-1, 1], granularity=30):
    x_1s = np.linspace(x1_lim[0], x1_lim[1], granularity)
    x_2s = np.linspace(x2_lim[0], x2_lim[1], granularity)
    x_1, x_2 = np.meshgrid(x_1s, x_2s, indexing='xy')
    points = np.hstack([x_1.reshape(-1, 1), x_2.reshape(-1, 1)])

    y, prob_y, f = sampler(points)

    f_contour = ax.contour(x_1, x_2, f.reshape(granularity, granularity), cmap=plt.get_cmap('autumn'))
    py_contour = ax.contourf(x_1, x_2, prob_y.reshape(granularity, granularity), cmap=plt.get_cmap('Purples'))

    # down sample to just n
    plot_ix = np.random.choice(points.shape[0], N)
    plot_x = points[plot_ix]
    plot_y = y[plot_ix]
    ax.scatter(plot_x[plot_y == 1, 0], plot_x[plot_y == 1, 1], marker='x', c='red', label='+1')
    ax.scatter(plot_x[plot_y == -1, 0], plot_x[plot_y == -1, 1], marker='x', c='green', label='-1')

    return f_contour, py_contour