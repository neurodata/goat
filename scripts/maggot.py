#%%
import time
from collections import namedtuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from pkg.data import load_adjacency, load_node_meta
from pkg.gmp import quadratic_assignment, quadratic_assignment_ot


def get_paired_inds(meta, check_in=True, pair_key="pair", pair_id_key="pair_id"):
    pair_meta = meta.copy()
    pair_meta = pair_meta[pair_meta["hemisphere"].isin(["L", "R"])]
    if check_in:
        pair_meta = pair_meta[pair_meta[pair_key].isin(pair_meta.index)]
    pair_group_size = pair_meta.groupby(pair_id_key).size()
    remove_pairs = pair_group_size[pair_group_size == 1].index
    pair_meta = pair_meta[~pair_meta[pair_id_key].isin(remove_pairs)]
    assert pair_meta.groupby(pair_id_key).size().min() == 2
    assert pair_meta.groupby(pair_id_key).size().max() == 2
    pair_meta.sort_values([pair_id_key, "hemisphere"], inplace=True)
    lp_inds = pair_meta[pair_meta["hemisphere"] == "L"]["inds"]
    rp_inds = pair_meta[pair_meta["hemisphere"] == "R"]["inds"]
    assert (
        meta.iloc[lp_inds][pair_id_key].values == meta.iloc[rp_inds][pair_id_key].values
    ).all()
    return lp_inds, rp_inds


SplitAdjacencies = namedtuple("SplitAdjacencies", ["ll", "rr", "lr", "rl"])


def bisect(adj, meta, check_in=True, pair_key="pair", pair_id_key="pair_id"):
    meta["inds"] = range(len(meta))
    lp_inds, rp_inds = get_paired_inds(meta)

    adjs = []
    adjs.append(adj[np.ix_(lp_inds, lp_inds)])
    adjs.append(adj[np.ix_(rp_inds, rp_inds)])
    adjs.append(adj[np.ix_(lp_inds, rp_inds)])
    adjs.append(adj[np.ix_(rp_inds, lp_inds)])
    split = SplitAdjacencies(*adjs)
    return split


meta = load_node_meta(dataset="maggot")
adj = load_adjacency(dataset="maggot", graph_type="G", nodelist=meta.index)
ll_adj, rr_adj, _, _ = bisect(adj, meta)


#%%


def compute_match_ratio(inds, correct_inds):
    matched = inds == correct_inds
    return np.mean(matched)


rows = []
for i in range(1):
    print(f"Shuffle: {i}")
    shuffle_inds = np.random.permutation(len(rr_adj))
    inv_shuffle_inds = np.argsort(shuffle_inds)
    shuffle_rr_adj = rr_adj[np.ix_(shuffle_inds, shuffle_inds)]
    currtime = time.time()
    res_vanilla = quadratic_assignment(
        ll_adj, shuffle_rr_adj, options=dict(maximize=True, tol=1e-5, maxiter=150),
    )
    elapsed = time.time() - currtime
    res_vanilla["method"] = "vanilla"
    res_vanilla["elapsed"] = elapsed
    res_vanilla["match_ratio"] = compute_match_ratio(
        res_vanilla["col_ind"], inv_shuffle_inds
    )
    res_vanilla["reg"] = "vanilla"
    print(res_vanilla["match_ratio"])
    rows.append(res_vanilla)

for reg in [700]:
    print(f"Reg: {reg}")
    currtime = time.time()
    res_ls = quadratic_assignment_ot(
        ll_adj,
        shuffle_rr_adj,
        options=dict(
            maximize=True, reg=reg, tol=1e-5, maxiter=150, thr=1e-2, grad_thresh=1e-10
        ),
    )
    elapsed = time.time() - currtime
    res_ls["reg"] = reg
    res_ls["elapsed"] = elapsed
    res_ls["method"] = "light-speed"
    res_ls["match_ratio"] = compute_match_ratio(res_ls["col_ind"], inv_shuffle_inds)
    rows.append(res_ls)
    print(res_ls["match_ratio"])
print("\n\n")

# compute for the "given" or "true" matching
obvfun = np.trace(ll_adj.T @ rr_adj)  # A.T @ I @ B @ I.T
res = {"method": "known", "match_ratio": 1, "fun": obvfun, "reg": "truth"}
rows.append(res)
results = pd.DataFrame(rows)
results[["method", "match_ratio", "fun", "reg"]]


#%%
import matplotlib.pyplot as plt
import seaborn as sns
from pkg.plot import set_theme
from pkg.io import savefig


def stashfig(name, **kwargs):
    savefig(name, foldername="light_speed", print_out=False, **kwargs)


def shade(ax, idx=0):
    ax.axvspan(idx - 0.5, idx + 0.5, facecolor="lightgrey", alpha=0.6, zorder=-1)


set_theme()
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
sns.stripplot(data=results, x="reg", y="match_ratio", ax=axs[0, 0])
sns.stripplot(data=results, x="reg", y="fun", ax=axs[0, 1])
sns.stripplot(data=results, x="reg", y="elapsed", ax=axs[1, 0])
sns.stripplot(data=results, x="reg", y="nit", ax=axs[1, 1])
n_unique = results["reg"].nunique()
for ax in axs.flat:
    shade(ax, idx=0)
    shade(ax, idx=n_unique - 1)
plt.tight_layout()
stashfig("maggot-gmot-comparison")
