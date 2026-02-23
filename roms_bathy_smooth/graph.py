"""Graph utilities for bathymetry smoothing.

Connected component detection and BFS neighborhood expansion
used by the LP heuristic decomposition.

Replaces: GRAPH_ConnectedComponent.m, SUB_StepNeighborhood.m
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as scipy_cc
from collections import deque


def connected_components(edges, n_vertices):
    """Find connected components of an undirected graph.

    Args:
        edges: Array of shape (n_edges, 2), pairs of vertex indices (0-based).
        n_vertices: Total number of vertices.

    Returns:
        labels: Array of shape (n_vertices,), component label for each vertex
                (1-based, matching MATLAB convention).
    """
    if len(edges) == 0:
        return np.arange(1, n_vertices + 1)

    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row), dtype=np.int8)
    adj = csr_matrix((data, (row, col)), shape=(n_vertices, n_vertices))

    n_comp, labels_0 = scipy_cc(adj, directed=False)
    return labels_0 + 1  # 1-based


def step_neighborhood(mask, i_center, j_center, kdist):
    """Find all sea points reachable within kdist steps on the grid.

    BFS flood-fill on the mask, returning all sea points within kdist
    grid steps of (i_center, j_center), excluding the center itself.

    Args:
        mask: Land/sea mask, shape (eta_rho, xi_rho). 1=sea, 0=land.
        i_center: Row index of center point.
        j_center: Column index of center point.
        kdist: Maximum number of steps.

    Returns:
        neighbors: Array of shape (n_neighbors, 2), each row is [i, j].
    """
    eta, xi = mask.shape
    visited = np.zeros((eta, xi), dtype=bool)
    visited[i_center, j_center] = True

    queue = deque([(i_center, j_center, 0)])
    neighbors = []
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    while queue:
        ci, cj, dist = queue.popleft()
        if dist > 0:
            neighbors.append([ci, cj])
        if dist < kdist:
            for di, dj in directions:
                ni, nj = ci + di, cj + dj
                if (0 <= ni < eta and 0 <= nj < xi
                        and mask[ni, nj] == 1 and not visited[ni, nj]):
                    visited[ni, nj] = True
                    queue.append((ni, nj, dist + 1))

    return np.array(neighbors, dtype=int).reshape(-1, 2) if neighbors else np.empty((0, 2), dtype=int)
