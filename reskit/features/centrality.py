import numpy as np
import scipy
import igraph as ig

def closeness_centrality(X):
    n_nodes = X.shape[0]
    A_inv = 1./X
    SPL = scipy.sparse.csgraph.dijkstra(A_inv, directed=False,
            unweighted=False)
    sum_distances_vector = np.sum(SPL, 1)
    cl_c = float(n_nodes - 1)/sum_distances_vector
    featurized_X = cl_c
    return featurized_X

def betweenness_centrality(X):
    n_nodes = X.shape[0]
    A_inv = 1./X
    G_inv = ig.Graph.Weighted_Adjacency(list(A_inv), mode="UNDIRECTED", attr="weight", loops=False)
    btw = np.array(G_inv.betweenness(weights='weight', directed=False))*2./((n_nodes-1)*(n_nodes-2))
    return btw

def eigenvector_centrality(X):
    G = ig.Graph.Weighted_Adjacency(list(X), mode="UNDIRECTED",
                attr="weight", loops=False)
    eigc = G.eigenvector_centrality(weights='weight', directed=False)
    return np.array(eigc)

def pagerank(X):
    G = ig.Graph.Weighted_Adjacency(list(X), mode="DIRECTED", attr="weight", loops=False)
    return np.array(G.pagerank(weights="weight"))

__all__ = ['closeness_centrality',
           'betweenness_centrality',
           'eigenvector_centrality',
           'pagerank']
