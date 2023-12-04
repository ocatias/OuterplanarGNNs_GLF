import copy
import csv
import os

import scipy.sparse as sp
from ogb.graphproppred import PygGraphPropPredDataset
import torch
from torch_geometric.datasets import ZINC
from torch_geometric.utils import get_laplacian, to_dense_adj, to_scipy_sparse_matrix
import torch_geometric.transforms as T
from torch.linalg import eig
from tqdm import tqdm

from Misc.config import config
from Misc.cyclic_adjacency_transform import CyclicAdjacencyTransform
import numpy as np
from scipy.linalg import null_space, solve_continuous_lyapunov

CAT = CyclicAdjacencyTransform(debug=False, spiderweb=False)
spiderCAT = CyclicAdjacencyTransform(debug=False, spiderweb=True)
node_remover = T.Compose([T.remove_isolated_nodes.RemoveIsolatedNodes()])


def symmetric_resistance(graph):
    laplacian = get_laplacian(graph.edge_index)
    dense_laplacian = to_dense_adj(edge_index=laplacian[0], edge_attr=laplacian[1])[0]
    eigenvalues = eig(dense_laplacian).eigenvalues
    assert eigenvalues.imag.sum() == 0.0
    eigenvalues = eigenvalues.real.sort()[0][1:]  # Take real part, sort, exclude smallest eigenvalue.
    resistance = graph.num_nodes * (1 / eigenvalues).sum()  # R = n * tr(L+)

    return resistance, resistance / graph.num_nodes


def directed_resistance(graph):
    # From https://arxiv.org/pdf/1310.5163.pdf
    if graph.num_nodes < 2:
        # Resistance of single nodes taken as zero.
        return np.array([0])

    laplacian = get_laplacian(graph.edge_index)
    dense_laplacian = to_dense_adj(edge_index=laplacian[0], edge_attr=laplacian[1])[0]
    n = dense_laplacian.shape[0]

    # Create subspace perpendicular to vector 1n.
    A = np.zeros((n, n))
    A[0] = np.ones(n)
    Q = null_space(A).T  # Ortonormal basis of nullspace, with basis as row vectors.

    # Projection matrix onto that subspace. -> np.matmul(Q, np.ones((n, 1))) should be ~0.
    # pi = np.identity(n) - (1/float(n)) * (np.matmul(np.ones((n, 1)), np.ones((n, 1)).T))

    L = Q.dot(dense_laplacian).dot(Q.T)  # Reduced laplacian.
    S = solve_continuous_lyapunov(L.astype(np.float64),
                                  np.identity(n - 1).astype(np.float64))  # Solution of Lyapunov equation.
    X = 2 * Q.T.dot(S).dot(Q)

    # Get resistances.
    R = np.zeros((n, n))
    for k, row in enumerate(X):
        for j, cell in enumerate(row):
            R[k][j] = X[k][k] + X[j][j] - 2 * X[k][j]

    return R


def compute_dataset_resistance(dataset, transform_func, resistance_func):
    mean_resistance = []
    max_resistance = []
    sum_resistance = []
    diameter = []
    total_components = 0
    total_components_before_removal = 0
    number_lone_nodes = 0

    for i, graph in enumerate(tqdm(dataset)):

        adj = to_scipy_sparse_matrix(graph.edge_index, num_nodes=graph.num_nodes)
        num_components_before_removal, component = sp.csgraph.connected_components(adj, connection="weak")
        total_components_before_removal += num_components_before_removal
        if transform_func is not None:
            graph = transform_func(graph)
        initial_nodes = graph.num_nodes
        graph = node_remover(graph)
        no_lone_nodes = graph.num_nodes
        lone_nodes = initial_nodes - no_lone_nodes
        number_lone_nodes += lone_nodes

        # Check for multiple components.
        adj = to_scipy_sparse_matrix(graph.edge_index, num_nodes=graph.num_nodes)
        num_components, component = sp.csgraph.connected_components(adj, connection="weak")
        _, count = np.unique(component, return_counts=True)
        total_components += num_components

        for i in range(0, num_components):
            subset = np.in1d(component, count.argsort()[i])
            graph_component = graph.clone().subgraph(torch.from_numpy(subset).to(torch.bool))
            R = resistance_func(graph_component)
            mean_resistance.append(R.mean())
            max_resistance.append(R.max())
            sum_resistance.append(R.sum())

            # Compute component diameter.
            adj = to_scipy_sparse_matrix(graph_component.edge_index,
                                         num_nodes=graph_component.num_nodes)
            if adj.size > 0:  # Else, the diameter is not defined.
                diam = np.max(sp.csgraph.shortest_path(sp.coo_array(adj).tocsr()))
                diameter.append(diam)

    print(f"\nTotal components before any transformation: {total_components_before_removal}")
    print(f"Total components after transformation and lone node removal: {total_components}\n")
    print(f"Number lone nodes: {number_lone_nodes}\n")

    return mean_resistance, sum_resistance, max_resistance, diameter, total_components


ogbg_datasets = ["ogbg-molbace", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbbbp",
                 "ogbg-molsider", "ogbg-moltoxcast", "ogbg-mollipo", "ogbg-molhiv"]

datasets = ["zinc"] + ogbg_datasets
splits = ["train", "test"]

for dataset_name in datasets:
    print(f"\n\n{dataset_name}")

    for split in splits:
        if dataset_name == "zinc":
            ds = ZINC(root=config.DATA_PATH, subset=True, split=split)
            ds_no_cat = copy.deepcopy(ds)
        elif dataset_name in ogbg_datasets:
            ds = PygGraphPropPredDataset(root=config.DATA_PATH, name=dataset_name)
            split_idx = ds.get_idx_split()
            ds = ds[split_idx[split]]
            ds_no_cat = copy.deepcopy(ds)
        else:
            raise "Not implemented"

        results_path = "Results/"
        digits = 3

        print("CAT")
        r_mean, r_sum, r_max, diam, total_components = compute_dataset_resistance(ds, CAT, directed_resistance)
        print("noCAT")
        sr_mean, sr_sum, sr_max, s_diam, s_total_components = compute_dataset_resistance(ds_no_cat, None, directed_resistance)

        header = ["method",
                  "avg_mean_res", "std_mean_res",
                  "avg_tot_res", "std_tot_res",
                  "avg_max_res", "std_max_res", "diam", "std_diam", "num_components"]
        average_cat = ["CAT",
                       np.round(np.mean(r_mean), digits), np.round(np.std(r_mean), digits),
                       np.round(np.mean(r_sum), digits), np.round(np.std(r_sum), digits),
                       np.round(np.mean(r_max), digits), np.round(np.std(r_max), digits),
                       np.round(np.mean(diam), digits), np.round(np.std(diam), digits),
                       total_components]
        average_no_cat = ["noCAT",
                          np.round(np.mean(sr_mean), digits), np.round(np.std(sr_mean), digits),
                          np.round(np.mean(sr_sum), digits), np.round(np.std(sr_sum), digits),
                          np.round(np.mean(sr_max), digits), np.round(np.std(sr_max), digits),
                          np.round(np.mean(s_diam), digits), np.round(np.std(s_diam), digits),
                          s_total_components]

        with open(os.path.join(results_path, dataset_name + split + ".csv"), 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(header)
            csvwriter.writerow(average_cat)
            csvwriter.writerow(average_no_cat)

        with open(os.path.join(results_path, dataset_name + split + "_all.csv"), 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(["r_mean"] + r_mean)
            csvwriter.writerow(["r_sum"] + r_sum)
            csvwriter.writerow(["r_max"] + r_max)
            csvwriter.writerow(["sr_mean"] + sr_mean)
            csvwriter.writerow(["sr_sum"] + sr_sum)
            csvwriter.writerow(["sr_max"] + sr_max)