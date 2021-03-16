#%%
from pathlib import Path

import graspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspy.embed import *
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import *
import pickle


data_dir = Path(".")
data_dir = data_dir / "Cook et al revised Supplementary Information"
matdir = data_dir / "SI 5 Connectome adjacency matrices Pedigo.xlsx"
celldir = "./nice_data/master_cells.csv"
n_verify = 1000
save_me = True
run_homologs = True
plot = True


def verify(n, cells, adj, original, error="error"):
    for i in range(n):
        rand_ind_out = np.random.randint(0, len(cells))
        rand_ind_in = np.random.randint(0, len(cells))
        in_cell = cells[rand_ind_in]
        out_cell = cells[rand_ind_out]
        saved_weight = adj[rand_ind_in, rand_ind_out]
        try:
            original_weight = original.loc[in_cell, out_cell]
            if saved_weight != original_weight:
                print(error)
                print("edge")
                print(out_cell)
                print(in_cell)
                print(saved_weight)
                print(original_weight)
        except:
            pass


def verify_undirected(n, cells, adj, original, error="error"):
    for i in range(n):
        rand_ind_out = np.random.randint(0, len(cells))
        rand_ind_in = np.random.randint(0, len(cells))
        in_cell = cells[rand_ind_in]
        out_cell = cells[rand_ind_out]
        saved_weight = adj[rand_ind_in, rand_ind_out]
        try:
            original_weight1 = original.loc[in_cell, out_cell]
            original_weight2 = original.loc[out_cell, in_cell]
            if saved_weight != original_weight1 and save_weight != original_weight2:
                print(error)
                print("edge")
                print(out_cell)
                print(in_cell)
                print(saved_weight)
                print(original_weight)
        except:
            pass


def load_df(matdir, sheet_name):
    df = pd.read_excel(matdir, sheet_name=sheet_name).fillna(0)
    outs = df.index.values
    ins = df.columns.values
    outs = upper_array(outs)
    ins = upper_array(ins)
    df.columns = ins
    df.index = outs
    return df


def upper_array(arr):
    upper = [u.upper() for u in arr]
    return np.array(upper)


def emmons_excel_to_df(matdir, sheet_name):
    df = load_df(matdir, sheet_name)
    # get the in / out cells
    # have to append some 0s to make into a square matrix
    outs = df.index.values
    ins = df.columns.values
    not_outs = np.setdiff1d(ins, outs)
    not_outs_df = pd.DataFrame(columns=df.columns)
    temp = np.empty((len(not_outs), len(ins)))
    temp[:] = 0
    not_outs_df = pd.DataFrame(temp)
    not_outs_df.columns = df.columns
    not_outs_df.index = not_outs
    df_full = pd.concat([df, not_outs_df])
    # reindex so that indices on matrices mean the same thing
    df_full = df_full.reindex(list(ins))
    return df_full


# get directed and undirected graphs
def convert_from_df(df, g):
    g = nx.from_pandas_adjacency(df, create_using=g)
    full_cells = np.array(list(g.nodes))
    A_full = nx.to_numpy_array(g, full_cells)
    A_self, self_inds = get_lcc(A_full, return_inds=True)
    self_inds = np.array(self_inds)
    self_cells = full_cells[self_inds]
    return A_full, A_self, full_cells, self_cells


#%% load data
# get the excels as data frames, and add the missing values
herm_chem_df_full = emmons_excel_to_df(matdir, sheet_name=0)
herm_gap_df_full = emmons_excel_to_df(matdir, sheet_name=2)  # this is "undirected" one
male_chem_df_full = emmons_excel_to_df(matdir, sheet_name=3)
male_gap_df_full = emmons_excel_to_df(matdir, sheet_name=5)  # this is "undirected" one

# load the originals for comparison just to verify the process worked
herm_chem_o = load_df(matdir, sheet_name=0)
herm_gap_o = load_df(matdir, sheet_name=2)
male_chem_o = load_df(matdir, sheet_name=3)
male_gap_o = load_df(matdir, sheet_name=5)

#%% chem
# hermaphrodite chem self LCC
g = nx.DiGraph()
herm_chem_A_full, herm_chem_A_self, herm_chem_full_cells, herm_chem_self_cells = convert_from_df(
    herm_chem_df_full, g
)
verify(n_verify, herm_chem_self_cells, herm_chem_A_self, herm_chem_o, "herm self")

# male chem self LCC
g = nx.DiGraph()
male_chem_A_full, male_chem_A_self, male_chem_full_cells, male_chem_self_cells = convert_from_df(
    male_chem_df_full, g
)
verify(n_verify, male_chem_self_cells, male_chem_A_self, male_chem_o, "male self")

# herm chem self LCC undirected
g = nx.Graph()
herm_chem_A_full_undirected, herm_chem_A_self_undirected, _, _ = convert_from_df(
    herm_chem_df_full, g
)

# male chem self LCC undirected
g = nx.Graph()
male_chem_A_full_undirected, male_chem_A_self_undirected, _, _ = convert_from_df(
    male_chem_df_full, g
)
#%% gap
# herm gap undirected
g = nx.Graph()
herm_gap_A_full_undirected, _, herm_gap_full_cells, _ = convert_from_df(
    herm_gap_df_full, g
)
verify_undirected(
    n_verify,
    herm_gap_full_cells,
    herm_gap_A_full_undirected,
    herm_gap_o,
    "herm gap full",
)

# male gap undirected
g = nx.Graph()
male_gap_A_full_undirected, _, male_gap_full_cells, _ = convert_from_df(
    male_gap_df_full, g
)
verify_undirected(
    n_verify,
    male_gap_full_cells,
    male_gap_A_full_undirected,
    male_gap_o,
    "male gap full",
)

#%% find some of the intersections of graphs
# load both originals as networkx
g = nx.DiGraph()
herm_chem_g = nx.from_pandas_adjacency(herm_chem_df_full, create_using=g)
g = nx.DiGraph()
male_chem_g = nx.from_pandas_adjacency(male_chem_df_full, create_using=g)
# use graspy to find the intersection of lccs
# this is called "multi" here
shared_gs, shared_cells = get_multigraph_intersect_lcc(
    [herm_chem_g, male_chem_g], return_inds=True
)
herm_chem_g_multi = shared_gs[0]
male_chem_g_multi = shared_gs[1]
herm_chem_A_multi = nx.to_numpy_array(herm_chem_g_multi, nodelist=shared_cells)
male_chem_A_multi = nx.to_numpy_array(male_chem_g_multi, nodelist=shared_cells)
is_fully_connected(herm_chem_A_multi)
is_fully_connected(herm_chem_g_multi)
# do the same as above but for undirected, should be the same in theory
g = nx.Graph()
herm_chem_g_undirected = nx.from_pandas_adjacency(herm_chem_df_full, create_using=g)
g = nx.Graph()
male_chem_g_undirected = nx.from_pandas_adjacency(male_chem_df_full, create_using=g)
shared_gs, shared_cells2 = get_multigraph_intersect_lcc(
    [herm_chem_g_undirected, male_chem_g_undirected], return_inds=True
)
herm_chem_g_multi_undirected = shared_gs[0]
male_chem_g_multi_undirected = shared_gs[1]
herm_chem_A_multi_undirected = nx.to_numpy_array(
    herm_chem_g_multi_undirected, nodelist=shared_cells
)
male_chem_A_multi_undirected = nx.to_numpy_array(
    male_chem_g_multi_undirected, nodelist=shared_cells
)

verify(n_verify, shared_cells, herm_chem_A_multi, herm_chem_o, "herm multi")
verify(n_verify, shared_cells, male_chem_A_multi, male_chem_o, "male multi")

# save out
if save_me:
    data_path = Path("./nice_data")
    data_names = [
        "herm_chem_A_full",
        "herm_chem_A_self",
        "herm_chem_A_multi",
        "herm_chem_A_full_undirected",
        "herm_chem_A_self_undirected",
        "herm_chem_A_multi_undirected",
        "male_chem_A_full",
        "male_chem_A_self",
        "male_chem_A_multi",
        "male_chem_A_full_undirected",
        "male_chem_A_self_undirected",
        "male_chem_A_multi_undirected",
        "herm_gap_A_full_undirected",
        "male_gap_A_full_undirected",
    ]
    var_names = [
        herm_chem_A_full,
        herm_chem_A_self,
        herm_chem_A_multi,
        herm_chem_A_full_undirected,
        herm_chem_A_self_undirected,
        herm_chem_A_multi_undirected,
        male_chem_A_full,
        male_chem_A_self,
        male_chem_A_multi,
        male_chem_A_full_undirected,
        male_chem_A_self_undirected,
        male_chem_A_multi_undirected,
        herm_gap_A_full_undirected,
        male_gap_A_full_undirected,
    ]
    if plot:
        for v in var_names:
            heatmap(v, transform="simple-nonzero")
    for i, save_name in enumerate(data_names):
        file_name = data_path / str(save_name + ".csv")
        np.savetxt(file_name, var_names[i], delimiter=",")

    cell_var_names = [
        herm_chem_full_cells,
        herm_chem_self_cells,
        male_chem_full_cells,
        male_chem_self_cells,
        shared_cells,
        herm_gap_full_cells,
        male_gap_full_cells,
    ]
    cell_save_names = [
        "herm_chem_full_cells",
        "herm_chem_self_cells",
        "male_chem_full_cells",
        "male_chem_self_cells",
        "chem_multi_cells",
        "herm_gap_full_cells",
        "male_gap_full_cells",
    ]
    for i, save_name in enumerate(cell_save_names):
        file_name = data_path / str(save_name + ".csv")
        np.savetxt(file_name, cell_var_names[i], fmt="%-1s", delimiter=",")

    ce_cells = dict(zip(cell_save_names, cell_var_names))
    ce_connect = dict(zip(data_names, var_names))

    with open("./nice_data/ce_cells.pickle", "wb") as f:
        pickle.dump(ce_cells, f)
    with open("./nice_data/ce_connect.pickle", "wb") as f:
        pickle.dump(ce_connect, f)
#%%
def get_match_name(name, side):
    loc = name.rfind("side")
    match_guess = list(name)
    if side == "L":
        opp = "R"
    if side == "R":
        opp = "L"
    match_guess[loc] = opp
    key = "".join(match_guess)
    return key


def check_match(name, all_names):
    l_loc = name.rfind("L")
    r_loc = name.rfind("R")
    locs = [l_loc, r_loc]
    sides = ["L", "R"]
    opp_sides = ["R", "L"]
    # if l_loc != -1 and r_loc != -1:
    #     # print("hmm")
    #     print(name)
    opp_loc_ind = np.argmax(locs)
    opp_type = opp_sides[opp_loc_ind]
    opp_loc = locs[opp_loc_ind]
    match_guess = list(name)
    match = ""
    side = ""
    if opp_loc != -1:
        match_guess[opp_loc] = opp_type
        match_guess = "".join(match_guess)
        if match_guess in all_names:
            match = match_guess
            if opp_type == "R":
                side = "left"
            if opp_type == "L":
                side = "right"
    return match, side


if run_homologs:
    cell_df = pd.read_csv(celldir)
    cell_df = cell_df.set_index("name")
    homologs = len(cell_df.index) * ["na"]
    cell_df["sidepaired"] = homologs
    cell_df["homolog"] = homologs
    all_cells = cell_df.index.values
    for name, row in cell_df.iterrows():
        match, side = check_match(name, all_cells)
        if match is not "":
            cell_df.loc[name, "sidepaired"] = side
            cell_df.loc[name, "homolog"] = match
    if save_me:
        cell_df.to_csv("./nice_data/master_cells.csv")


#%%
