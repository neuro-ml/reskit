import numpy as np
import pandas as pd
import scipy
import igraph as ig
import networkx as nx


def degrees(data):
    if data['X'].ndim == 3:
        data['degrees'] = np.sum(data['X'], axis=1)
    elif data['X'].ndim == 2:
        data['degrees'] = np.sum(data['X'], axis=1)
    else:
        raise ValueError('Provide array of valid shape: (number_of_matrices, size, size). ')

    return data


def closeness_centrality(data):
    n_nodes = data['X'].shape[0]
    A_inv = 1./data['X']
    SPL = scipy.sparse.csgraph.dijkstra(A_inv, directed=False,
            unweighted=False)
    sum_distances_vector = np.sum(SPL, 1)
    data['closeness_centrality'] = float(n_nodes - 1)/sum_distances_vector
    return data

def betweenness_centrality(data):
    n_nodes = data['X'].shape[0]
    A_inv = 1./data['X']
    G_inv = ig.Graph.Weighted_Adjacency(list(A_inv), mode="UNDIRECTED", attr="weight", loops=False)
    data['betweenness_centrality'] = np.array(G_inv.betweenness(weights='weight', directed=False))*2./((n_nodes-1)*(n_nodes-2))
    return data

def eigenvector_centrality(data):
    G = ig.Graph.Weighted_Adjacency(list(data['X']), mode="UNDIRECTED",
                attr="weight", loops=False)
    data['eigenvector_centrality'] = G.eigenvector_centrality(weights='weight', directed=False)
    data['eigenvector_centrality'] = np.array(data['eigenvector_centrality'])
    return data

def pagerank(data):
    G = ig.Graph.Weighted_Adjacency(list(data['X']), mode="DIRECTED", attr="weight", loops=False)
    data['pagerank'] = np.array(G.pagerank(weights="weight"))
    return data


def clustering_coefficient(data):
    Gnx = nx.from_numpy_matrix(data['X'])
    data['clustering_coefficient'] = list(nx.clustering(Gnx, weight='weight').values())
    data['clustering_coefficient'] = np.array(data['clustering_coefficient'])
    return data

def triangles(data):
    clust = clustering_coefficient(data['X'])

    G = ig.Graph.Weighted_Adjacency(list(data['X']), mode="UNDIRECTED",
            attr="weight", loops=False)
    non_weighted_degrees = np.array(G.degree())
    non_weighted_deg_by_deg_minus_one = np.multiply(non_weighted_degrees,
            (non_weighted_degrees - 1))
    data['triangles'] = np.multiply(np.array(clust),
            np.array(non_weighted_deg_by_deg_minus_one, dtype = float))/2.
    return data


def efficiency(data):
    A_inv = 1./data['X']
    SPL = scipy.sparse.csgraph.dijkstra(A_inv, directed=False, unweighted=False)
    inv_SPL_with_inf = 1./SPL
    inv_SPL_with_nan = inv_SPL_with_inf.copy()
    inv_SPL_with_nan[np.isinf(inv_SPL_with_inf)]=np.nan
    data['efficiency'] = np.nanmean(inv_SPL_with_nan, 1)
    return data


def bag_of_edges(data, SPL=None, symmetric = True, return_df = False, offset = 1):
    size = data['X'].shape[1]
    if symmetric:
        indices = np.triu_indices(size, k = offset)
    else:
        grid = np.indices(data['X'].shape[1:])
        indices = (grid[0].reshape(-1), grid[1].reshape(-1))
    if len(data['X'].shape) == 3:
        data['bag_of_edges'] = data['X'][:, indices[0], indices[1]]
    elif len(X.shape) == 2:
        data['bag_of_edges'] = X[indices[0], indices[1]]
    else:
        raise ValueError('Provide array of valid shape: (number_of_matrices, size, size).')
    if return_df:
        col_names = ['edge_' + str(i) + '_' + str(j) for i,j in zip(indices[0], indices[1])]
        featurized_X = pd.DataFrame(data['bag_of_edges'], columns=col_names)
        return featurized_X
    return data


__all__ = ['bag_of_edges',
           'closeness_centrality',
           'betweenness_centrality',
           'eigenvector_centrality',
           'pagerank',
           'degrees',
           'clustering_coefficient',
           'triangles',
           'efficiency']
