"""
Node Embeddings and Link Prediction from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - karate_club_graph
import torch

KARATE_EDGES = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
    (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
    (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
    (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
    (3, 7), (3, 12), (3, 13),
    (4, 6), (4, 10),
    (5, 6), (5, 10), (5, 16),
    (6, 16),
    (8, 30), (8, 32), (8, 33),
    (9, 33),
    (13, 33),
    (14, 32), (14, 33),
    (15, 32), (15, 33),
    (18, 32), (18, 33),
    (19, 33),
    (20, 32), (20, 33),
    (22, 32), (22, 33),
    (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
    (24, 25), (24, 27), (24, 31),
    (25, 31),
    (26, 29), (26, 33),
    (27, 33),
    (28, 31), (28, 33),
    (29, 32), (29, 33),
    (30, 32), (30, 33),
    (31, 32), (31, 33),
    (32, 33)
]

KARATE_LABELS = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 1, 1, 0, 0, 1, 0,
    1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1
]


def karate_club_graph():
    # Convert the undirected edge list into a tensor containing
    # all original directions first, followed by all reverse directions.
    edges = torch.tensor(KARATE_EDGES, dtype=torch.long)

    forward_edges = edges.t()
    reverse_edges = edges[:, [1, 0]].t()

    edge_index = torch.cat([forward_edges, reverse_edges], dim=1)

    labels = torch.tensor(KARATE_LABELS, dtype=torch.long)

    return edge_index, labels


def undirected_edge_set(edge_index):
    # Collect each edge as an ordered (min, max) pair so that
    # (u, v) and (v, u) are treated as the same undirected edge.
    edges = set()

    for u, v in edge_index.t().tolist():
        if u == v:
            continue

        edges.add((min(u, v), max(u, v)))

    return edges

# Step 2 - build_adjacency_lists
def build_adjacency_lists(edge_index, n):
    # Use sets temporarily so duplicate edges are removed automatically.
    neighbours = [set() for _ in range(n)]

    for u, v in edge_index.t().tolist():
        # Ignore self-loops.
        if u == v:
            continue

        # Treat every edge as undirected.
        neighbours[u].add(v)
        neighbours[v].add(u)

    # Convert each set into a sorted list.
    adj = [sorted(node_neighbours) for node_neighbours in neighbours]

    return adj


def degree_vector(adj):
    # The degree of each node is simply the length of its neighbour list.
    return torch.tensor([len(neighbours) for neighbours in adj], dtype=torch.long)

