import numpy as np
import pandas as pd
import scipy
import igraph as ig
import networkx as nx


def degrees(X):
    """
    Degree graph metric.

    Parameters
    ----------
    X : numpy matrix
        Adjacency matrix of a graph.

    Returns
    -------
    degrees : numpy array
        Degrees of nodes for the graph.
    """
    if X.ndim == 3:
        degrees = np.sum(X, axis=1)
    elif X.ndim == 2:
        degrees = np.sum(X, axis=1)
    else:
        raise ValueError(
            'Provide array of valid shape: (number_of_matrices, size, size). ')
    return degrees


def closeness_centrality(X):
    """
    Closeness centrality graph metric.

    Parameters
    ----------
    X : numpy matrix
        Adjacency matrix of a graph.

    Returns
    -------
    cl_c : numpy array
        Closeness centralities of nodes for the graph.
    """
    n_nodes = X.shape[0]
    A_inv = 1. / (X + 1.00000000e-99)
    SPL = scipy.sparse.csgraph.dijkstra(A_inv, directed=False,
                                        unweighted=False)
    sum_distances_vector = np.sum(SPL, 1)
    cl_c = float(n_nodes - 1) / sum_distances_vector
    return cl_c


def betweenness_centrality(X):
    """
    Betweenness centrality graph metric.

    Parameters
    ----------
    X : numpy matrix
        Adjacency matrix of a graph.

    Returns
    -------
    btw : numpy array
        Betweenness centralities of nodes for the graph.
    """
    n_nodes = X.shape[0]
    A_inv = 1. / (X + 1.00000000e-99)
    G_inv = ig.Graph.Weighted_Adjacency(
        list(A_inv), mode="UNDIRECTED", attr="weight", loops=False)
    btw = np.array(G_inv.betweenness(weights='weight',
                                          directed=False)) * 2. / ((n_nodes - 1) * (n_nodes - 2))
    return btw


def eigenvector_centrality(X):
    """
    Eigenvector centrality graph metric.

    Parameters
    ----------
    X : numpy matrix
        Adjacency matrix of a graph.

    Returns
    -------
    eigc : numpy array
        Eigenvector centralities of nodes for the graph.
    """
    G = ig.Graph.Weighted_Adjacency(list(X), mode="UNDIRECTED",
                                    attr="weight", loops=False)
    eigcX = G.eigenvector_centrality(weights='weight', directed=False)
    eigc = np.array(eigcX)
    return eigc


def pagerank(X):
    """
    Pagerank graph metric.

    Parameters
    ----------
    X : numpy matrix
        Adjacency matrix of a graph.

    Returns
    -------
    pgrnk : numpy array
        Pagerank vector for the graph.
    """
    G = ig.Graph.Weighted_Adjacency(
        list(X), mode="DIRECTED", attr="weight", loops=False)
    pgrnk = np.array(G.pagerank(weights="weight"))
    return pgrnk


def clustering_coefficient(X):
    """
    Clustering coefficient graph metric.

    Parameters
    ----------
    X : numpy matrix
        Adjacency matrix of a graph.

    Returns
    -------
    clst_geommean : numpy array
        Clustering coefficients of nodes for the graph.
    """
    Gnx = nx.from_numpy_matrix(X)
    clst_geommeanX = list(nx.clustering(Gnx, weight='weight').values())
    clst_geommean = np.array(clst_geommeanX)
    return clst_geommean


def triangles(X):
    """
    Triangles graph metric.

    Parameters
    ----------
    X : numpy matrix
        Adjacency matrix of a graph.

    Returns
    -------
    tr : numpy array
        Triangles of nodes for the graph.
    """
    clust = clustering_coefficient(X)

    G = ig.Graph.Weighted_Adjacency(list(X), mode="UNDIRECTED",
                                    attr="weight", loops=False)
    non_weighted_degrees = np.array(G.degree())
    non_weighted_deg_by_deg_minus_one = np.multiply(
        non_weighted_degrees, (non_weighted_degrees - 1))
    tr = np.multiply(
        np.array(clustX),
        np.array(
            non_weighted_deg_by_deg_minus_one,
            dtype=float)) / 2.
    return tr


def efficiency(X):
    """
    Efficiency graph metric.

    Parameters
    ----------
    X : numpy matrix
        Adjacency matrix of a graph.

    Returns
    -------
    efs : numpy array
        Efficiency of nodes for the graph.
    """
    A_inv = 1. / (X + 1.00000000e-99)
    SPL = scipy.sparse.csgraph.dijkstra(
        A_inv, directed=False, unweighted=False)
    inv_SPL_with_inf = 1. / (SPL + 1.00000000e-99)
    inv_SPL_with_nan = inv_SPL_with_inf.copy()
    inv_SPL_with_nan[np.isinf(inv_SPL_with_inf)] = np.nan
    efs = np.nanmean(inv_SPL_with_nan, 1)
    return efs


def bag_of_edges(X, SPL=None, symmetric=True, return_df=False, offset=1):
    """
    Bag of edges graph metric.

    Parameters
    ----------
    X : numpy matrix
        Adjacency matrix of a graph.

    Returns
    -------
    bag_of_edges : numpy array
        Bag of edges for the graph.
    """
    size = X.shape[1]
    if symmetric:
        indices = np.triu_indices(size, k=offset)
    else:
        grid = np.indices(X.shape[1:])
        indices = (grid[0].reshape(-1), grid[1].reshape(-1))
    if len(X.shape) == 3:
        bag_of_edges = X[:, indices[0], indices[1]]
    elif len(X.shape) == 2:
        bag_of_edges = X[indices[0], indices[1]]
    else:
        raise ValueError(
            'Provide array of valid shape: (number_of_matrices, size, size).')
    if return_df:
        col_names = ['edge_' + str(i) + '_' + str(j)
                     for i, j in zip(indices[0], indices[1])]
        featurized_X = pd.DataFrame(bag_of_edges, columns=col_names)
        return featurized_X
    return bag_of_edges


__all__ = ['bag_of_edges',
           'closeness_centrality',
           'betweenness_centrality',
           'eigenvector_centrality',
           'pagerank',
           'degrees',
           'clustering_coefficient',
           'triangles',
           'efficiency']
