import numpy as np
import networkx as nx
import igraph as ig

def clustering_coefficient(X):
    Gnx = nx.from_numpy_matrix(X)
    clst_geommean = list(nx.clustering(Gnx, weight='weight').values())
    clst_geommean
    return np.array(clst_geommean)

def triangles(X):
    clust = clustering_coefficient(X)

    G = ig.Graph.Weighted_Adjacency(list(X), mode="UNDIRECTED",
            attr="weight", loops=False)
    non_weighted_degrees = np.array(G.degree())
    non_weighted_deg_by_deg_minus_one = np.multiply(non_weighted_degrees,
            (non_weighted_degrees - 1))
    tr = np.multiply(np.array(clust),
            np.array(non_weighted_deg_by_deg_minus_one, dtype = float))/2.
    return tr

__all__ = ['clustering_coefficient',
           'triangles']
