import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_observations(plt_axes, observations, color):
    plt_axes.plot(observations[:,0], observations[:,1], color=color, linestyle="None", marker="+")

def plot_mean_variance(plt_axes, mean, variance, color, scale=1.0):
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

# begin read datas
observations = np.loadtxt("data10000.txt", delimiter=",")
obs_length = len(observations)

#define constants
K = 2
dim = len(observations[0])
T = len(observations)#number of a sequence length
obs_mean = observations.mean(axis=0)
obs_cov = np.cov(observations, rowvar=0, bias=1)
colors = ["#ff0000", "#00ff00"]
np.random.seed(0)

print "initialize hyper parameters"
#define hyper parameters and initialize
alpha = np.ones(K) * 5
p = np.array(list(obs_mean) * K).reshape([K, dim])
C = np.array(list(obs_cov) * K).reshape([K, dim, dim])
df = [dim] * K
psi = np.array(list(np.identity(dim) * 10) * K).reshape([K, dim, dim])

print "initialize local parameters"
#define parameters and initialize
pi = stats.dirichlet.rvs(alpha)[0]
mu = np.empty([K, dim])
Sigma = np.empty([K, dim, dim])
for k in range(K):
    mu[k] = stats.multivariate_normal.rvs(mean=p[k], cov=C[k])
    Sigma[k] = stats.invwishart.rvs(df=df[k], scale=psi[k])

#define variables
hidden_state = np.empty(T, dtype=np.uint8)

#define tmp variables
vc = np.zeros(K)
p_yi = np.empty([T, K])
hidden_state_count = np.zeros(K, dtype=np.uint8)

#define plot objects
fig = plt.figure()
ax = plt.axes()

#start iterations
while True:

    #calculate phase
    hidden_state_count *= 0
    for k in range(K):
        p_yi[:,k] = stats.multivariate_normal.pdf(observations, mean=mu[k], cov=Sigma[k])
    p_yi *= pi
    #resampling labels
    for t in range(T):
        vc[:] = p_yi[t]
        vc /= vc.sum()
        hidden_state[t] = np.random.choice(K, p=vc)
        hidden_state_count[hidden_state[t]] += 1

    #resampling parameters pahse
    #resampling pi
    pi = stats.dirichlet.rvs(alpha + hidden_state_count)[0]

    #resampling mu_k and Sigma_k and plot
    ax.cla()#clear axes
    for k in range(K):
        obs = observations[hidden_state == k]
        N = len(obs)
        sum_obs = obs.sum(axis=(0,))
        if N == 0:
            obs = np.zeros(dim)
            sum_obs = np.zeros(dim)

        #resampling Sigma_k
        obs_mu = np.matrix(obs - mu[k])
        psi_hat = obs_mu.T.dot(obs_mu) + psi[k]
        df_hat = N + df[k]
        Sigma[k] = stats.invwishart.rvs(df=df_hat, scale=psi_hat)

        #resampling mu_k
        C_hat = np.array((N * np.matrix(Sigma[k]).I + np.matrix(C[k]).I).I)
        p_hat = np.array((sum_obs.dot(np.matrix(Sigma[k]).I) + p[k].dot(np.matrix(C[k]).I)).dot(C_hat))[0]
        mu[k] = stats.multivariate_normal.rvs(mean=p_hat, cov=C_hat)

        #plot
        if N != 0:
            plot_observations(ax, obs, colors[k])
            plot_mean_variance(ax, mu[k], Sigma[k], colors[k])

    plt.pause(0.2)
