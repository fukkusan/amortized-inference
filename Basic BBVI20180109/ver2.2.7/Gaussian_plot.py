import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_observations(plt_axes, observations, color):
    plt_axes.plot(observations[:,0], observations[:,1], color=color, linestyle="None", marker="+")

def plot_mean_variance(plt_axes, mean, variance, color, scale=1.0):
    plt_axes.set_xlim([-5.0, 5.0])
    plt_axes.set_ylim([-5.0, 5.0])
    la, v = np.linalg.eig(variance)
    #la = np.sqrt(la)
    theta = -np.arctan(v[0,1]/v[0,0])
    x_lim = plt_axes.get_xlim()
    y_lim = plt_axes.get_ylim()
    x_width = abs(x_lim[0] - x_lim[1]) / 50.0
    y_width = abs(y_lim[0] - y_lim[1]) / 50.0
    ell = patches.Ellipse(
        xy=(mean[0],mean[1]),
        width=la[0]*scale/x_width,
        height=la[1]*scale/y_width,
        angle=theta * 180.0 / np.pi,
        ec=color,
        fill=False
        )
    plt_axes.plot(mean[0], mean[1], color=color, linestyle="None", marker="o")
    plt_axes.add_patch(ell)

