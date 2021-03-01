#%% [markdown]
# # Flow ranking and hypothesis testing
# TODO: explain the goal of finding a latent ordering, comparing between graphs

#%% [markdown]
# TODO: explain some of the math behind spring rank/signal flow

#%%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm

import SpringRank as sr
from giskard.plot import histplot
from pkg.flow import estimate_spring_rank_P
from pkg.io import savefig
from pkg.plot import set_theme
from src.visualization import adjplot

root_path = "/Users/bpedigo/JHU_code/maggot"
cwd = os.getcwd()
if cwd != root_path:
    os.chdir(root_path)


set_theme(font_scale=1.5)

rng = np.random.default_rng(seed=8888)


def stashfig(name, **kwargs):
    savefig(name, foldername="flow_rank_explain", print_out=False, **kwargs)


#%% [markdown]
# ## Creating latent "ranks" or "orderings"
# Here I sample some latent ranks that we'll use for simulations, this distribution came
# from the original paper.
#%%

colors = sns.color_palette("deep", desat=1)
palette = dict(zip(range(3), colors))

n_per_group = 100  # 34 in the paper
ones = np.ones(n_per_group, dtype=int)

X1 = rng.normal(-4, np.sqrt(2), size=n_per_group)
X2 = rng.normal(0, np.sqrt(1 / 2), size=n_per_group)
X3 = rng.normal(4, 1, size=n_per_group)
X = np.concatenate((X1, X2, X3))
labels = np.concatenate((0 * ones, 1 * ones, 2 * ones))

# sort to help visualize
sort_inds = np.argsort(-X)
X = X[sort_inds]
labels = labels[sort_inds]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(x=X, hue=labels, palette=palette, bins=50, stat="density", ax=ax)
sns.rugplot(
    x=X,
    hue=labels,
    palette=palette,
    height=0.05,
    legend=False,
    ax=ax,
    expand_margins=True,
)
stashfig("rank-distribution")

#%% [markdown]
# ## A distribution from the latent ranks
# Using the ranks, we can create a distribution from which to sample graphs. Here I plot
# the matrix of edge probabilities $P$ and an adjacency matrix $A$ from it.
#%%
k = 15
beta = 5


def construct_spring_rank_P(ranks, beta, degree):
    H = ranks[:, None] - ranks[None, :] - 1
    H = np.multiply(H, H)
    H *= 0.5
    P = np.exp(-beta * H)
    P /= np.mean(P) * len(P)
    P *= degree
    # TODO not sure this matches the paper exactly but makes sense to me
    return P


P = construct_spring_rank_P(X, beta, k)
A = rng.poisson(P)

fig, axs = plt.subplots(1, 2, figsize=(15, 7.5))
ax = axs[0]
adjplot(P, ax=ax, title=r"$P$", cbar=False)
ax = axs[1]
adjplot(A, ax=ax, title=r"$A$", color="darkred", plot_type="scattermap", sizes=(2, 5))
stashfig("p-and-adj")

#%% [markdown]
# If we change the parameters to be point masses for the 3 different groups, we get
# a specific kind of feedforward SBM model.
#%%
n_per_group = 100
X1 = np.ones(n_per_group)
X2 = np.ones(n_per_group) * 0
X3 = np.ones(n_per_group) * -1
X = np.concatenate((X1, X2, X3))
labels = np.concatenate((0 * ones, 1 * ones, 2 * ones))

k = 20
beta = 2

P = construct_spring_rank_P(X, beta, k)
A = rng.poisson(P)

fig, axs = plt.subplots(1, 2, figsize=(15, 7.5))
ax = axs[0]
adjplot(P, ax=ax, title=r"$P$", cbar=False)
ax = axs[1]
adjplot(A, ax=ax, title=r"$A$", color="darkred", plot_type="scattermap", sizes=(2, 5))
stashfig("p-and-adj-point-mass")

#%%

shuffle_inds = np.random.permutation(len(A))
shuffle_A = A[np.ix_(shuffle_inds, shuffle_inds)]
adjplot(
    shuffle_A,
    # title=r"$A$",
    color="darkred",
    plot_type="scattermap",
    sizes=(5, 5),
)
stashfig("shuffled_A")

ranks = sr.get_ranks(shuffle_A)
sort_inds = np.argsort(ranks)[::-1]
sort_A = shuffle_A[np.ix_(sort_inds, sort_inds)]
adjplot(
    sort_A,
    # title=r"$A$",
    color="darkred",
    plot_type="scattermap",
    sizes=(5, 5),
)
stashfig("sorted_A")

#%%

# set_theme(font_scale=1.5)
fig, axs = plt.subplots(1, 2, figsize=(5, 6), sharey=True)
ranks = sr.get_ranks(A)
ax = axs[0]
sns.scatterplot(y=ranks, x=np.zeros(len(ranks)), marker="_", linewidth=1, ax=ax)
ax.set(ylabel="Node flow score", yticks=[], xticks=[])
ax.spines["bottom"].set_visible(False)
ax = axs[1]
sns.histplot(y=ranks, kde=False, bins=30)
ax.set(yticks=[], xticks=[], xlabel="")
ax.spines["bottom"].set_visible(False)
stashfig("flow-score")
# %%
