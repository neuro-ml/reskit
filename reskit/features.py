import numpy as np
import pandas as pd
import scipy
import igraph as ig
import networkx as nx


def degrees(data):
    """
    Degree graph metric.

    Parameters
    ----------
    data : dict
        Dictionary with an item, which has key ``X``.

    Returns
    -------
    data : dict
        Dictionary with an itme, where stored features in key ``degrees``.
    """
    if data['X'].ndim == 3:
        data['degrees'] = np.sum(data['X'], axis=1)
    elif data['X'].ndim == 2:
        data['degrees'] = np.sum(data['X'], axis=1)
    else:
        raise ValueError(
            'Provide array of valid shape: (number_of_matrices, size, size). ')
    return data


def closeness_centrality(data):
    """
    Closeness centrality graph metric.

    Parameters
    ----------
    data : dict
        Dictionary with an item, which has key ``X``.

    Returns
    -------
    data : dict
        Dictionary with an itme, where stored features in key
        ``closeness_centrality``.
    """
    cl_c = []
    for X in data['X']:
        n_nodes = X.shape[0]
        A_inv = 1. / (X + 1.00000000e-99)
        SPL = scipy.sparse.csgraph.dijkstra(A_inv, directed=False,
                                            unweighted=False)
        sum_distances_vector = np.sum(SPL, 1)
        cl_c.append(float(n_nodes - 1) / sum_distances_vector)
    data['closeness_centrality'] = np.array(cl_c)
    return data


def betweenness_centrality(data):
    """
    Betweenness centrality graph metric.

    Parameters
    ----------
    data : dict
        Dictionary with an item, which has key ``X``.

    Returns
    -------
    data : dict
        Dictionary with an itme, where stored features in key
        ``betweenness_centrality``.
    """
    btw = []
    for X in data['X']:
        n_nodes = X.shape[0]
        A_inv = 1. / (X + 1.00000000e-99)
        G_inv = ig.Graph.Weighted_Adjacency(
            list(A_inv), mode="UNDIRECTED", attr="weight", loops=False)
        btw.append(np.array(G_inv.betweenness(weights='weight',
                                              directed=False)) * 2. / ((n_nodes - 1) * (n_nodes - 2)))
    data['betweenness_centrality'] = np.array(btw)
    return data


def eigenvector_centrality(data):
    """
    Eigenvector centrality graph metric.

    Parameters
    ----------
    data : dict
        Dictionary with an item, which has key ``X``.

    Returns
    -------
    data : dict
        Dictionary with an itme, where stored features in key
        ``eigenvector_centrality``.
    """
    eigc = []
    for X in data['X']:
        G = ig.Graph.Weighted_Adjacency(list(X), mode="UNDIRECTED",
                                        attr="weight", loops=False)
        eigcX = G.eigenvector_centrality(weights='weight', directed=False)
        eigc.append(np.array(eigcX))
    data['eigenvector_centrality'] = np.array(eigc)
    return data


def pagerank(data):
    """
    Pagerank graph metric.

    Parameters
    ----------
    data : dict
        Dictionary with an item, which has key ``X``.

    Returns
    -------
    data : dict
        Dictionary with an itme, where stored features in key ``pagerank``.
    """
    pgrnk = []
    for X in data['X']:
        G = ig.Graph.Weighted_Adjacency(
            list(X), mode="DIRECTED", attr="weight", loops=False)
        pgrnk.append(np.array(G.pagerank(weights="weight")))
    data['pagerank'] = np.array(pgrnk)
    return data


def clustering_coefficient(data):
    """
    Clustering coefficient graph metric.

    Parameters
    ----------
    data : dict
        Dictionary with an item, which has key ``X``.

    Returns
    -------
    data : dict
        Dictionary with an itme, where stored features in key
        ``clustering_coefficient``.
    """
    clst_geommean = []
    for X in data['X']:
        Gnx = nx.from_numpy_matrix(X)
        clst_geommeanX = list(nx.clustering(Gnx, weight='weight').values())
        clst_geommean.append(np.array(clst_geommeanX))
    data['clustering_coefficient'] = np.array(clst_geommean)
    return data


def triangles(data):
    """
    Triangles graph metric.

    Parameters
    ----------
    data : dict
        Dictionary with an item, which has key ``X``.

    Returns
    -------
    data : dict
        Dictionary with an itme, where stored features in key ``triangles``.
    """
    clust = clustering_coefficient(data)['clustering_coefficient']
    tr = []

    for X, clustX in zip(data['X'], clust):
        G = ig.Graph.Weighted_Adjacency(list(X), mode="UNDIRECTED",
                                        attr="weight", loops=False)
        non_weighted_degrees = np.array(G.degree())
        non_weighted_deg_by_deg_minus_one = np.multiply(
            non_weighted_degrees, (non_weighted_degrees - 1))
        tr.append(
            np.multiply(
                np.array(clustX),
                np.array(
                    non_weighted_deg_by_deg_minus_one,
                    dtype=float)) /
            2.)
    data['triangles'] = np.array(tr)
    return data


def efficiency(data):
    """
    Efficiency graph metric.

    Parameters
    ----------
    data : dict
        Dictionary with an item, which has key ``X``.

    Returns
    -------
    data : dict
        Dictionary with an itme, where stored features in key ``efficiency``.
    """
    efs = []
    for X in data['X']:
        A_inv = 1. / (X + 1.00000000e-99)
        SPL = scipy.sparse.csgraph.dijkstra(
            A_inv, directed=False, unweighted=False)
        inv_SPL_with_inf = 1. / (SPL + 1.00000000e-99)
        inv_SPL_with_nan = inv_SPL_with_inf.copy()
        inv_SPL_with_nan[np.isinf(inv_SPL_with_inf)] = np.nan
        efs.append(np.nanmean(inv_SPL_with_nan, 1))
    data['efficiency'] = np.array(efs)
    return data


def bag_of_edges(data, SPL=None, symmetric=True, return_df=False, offset=1):
    """
    Bag of edges graph metric.

    Parameters
    ----------
    data : dict
        Dictionary with an item, which has key ``X``.

    Returns
    -------
    data : dict
        Dictionary with an itme, where stored features in key ``bag_of_edges``.
    """
    size = data['X'].shape[1]
    if symmetric:
        indices = np.triu_indices(size, k=offset)
    else:
        grid = np.indices(data['X'].shape[1:])
        indices = (grid[0].reshape(-1), grid[1].reshape(-1))
    if len(data['X'].shape) == 3:
        data['bag_of_edges'] = data['X'][:, indices[0], indices[1]]
    elif len(X.shape) == 2:
        data['bag_of_edges'] = X[indices[0], indices[1]]
    else:
        raise ValueError(
            'Provide array of valid shape: (number_of_matrices, size, size).')
    if return_df:
        col_names = ['edge_' + str(i) + '_' + str(j)
                     for i, j in zip(indices[0], indices[1])]
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
