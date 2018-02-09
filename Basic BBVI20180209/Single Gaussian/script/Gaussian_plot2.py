import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import patches

def rad2deg(rad):
    return rad * 180.0 / np.pi

def plot_mean(axes, mean, color=None, marker="o", label=None):
    axes.plot(mean[0], mean[1], marker=marker, linestyle="", color=color, zorder=2, label=label)

def plot_covariance(axes, mean, covariance, scale=3.0, color=None, label=None):
    la, v = np.linalg.eig(covariance)
    std = np.sqrt(la)
    angle = rad2deg(np.arctan2(v[1,0], v[0,0]))

    e = patches.Ellipse((mean[0], mean[1]), 2.0*std[0]*scale, 2.0*std[1]*scale, angle=angle, linewidth=1, fill=False, color=color, zorder=2, label=label)

    axes.add_patch(e)

def plot_datas(axes, datas, color=None, marker="+", label=None):
    plot_datas_xy(axes, datas[:,0], datas[:,1], color=color, marker=marker)

def plot_datas_xy(axes, x, y, color=None, marker="+", label=None):
    axes.plot(x, y, marker=marker, linestyle="",  color=color, zorder=1, label=label)
