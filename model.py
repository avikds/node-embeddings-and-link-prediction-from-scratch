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

# Step 3 - sbm_graph
def sbm_graph(sizes, p_in, p_out, seed):
    # Assign each node to a block. Nodes are ordered block-by-block.
    blocks = torch.repeat_interleave(
        torch.arange(len(sizes), dtype=torch.long),
        torch.tensor(sizes, dtype=torch.long),
    )

    n = int(blocks.numel())

    # Use one generator and one n x n uniform draw, as required.
    g = torch.Generator().manual_seed(seed)
    rand = torch.rand((n, n), generator=g)

    # Probability of an edge depends on whether the two nodes
    # belong to the same block.
    same_block = blocks.unsqueeze(0) == blocks.unsqueeze(1)
    probabilities = torch.where(
        same_block,
        torch.tensor(p_in, dtype=rand.dtype),
        torch.tensor(p_out, dtype=rand.dtype),
    )

    # Keep only sampled edges in the strict upper triangle.
    upper_triangle = torch.triu(torch.ones((n, n), dtype=torch.bool), diagonal=1)
    sampled = upper_triangle & (rand < probabilities)

    # Extract all (u, v) pairs with u < v in row-major order.
    u, v = torch.nonzero(sampled, as_tuple=True)

    forward_edges = torch.stack([u, v], dim=0)
    reverse_edges = torch.stack([v, u], dim=0)

    # All upper-triangle edges first, followed by all reverses.
    edge_index = torch.cat([forward_edges, reverse_edges], dim=1)

    return edge_index, blocks


def sbm_features(blocks, dim, noise, seed):
    n = int(blocks.numel())
    num_blocks = int(blocks.max().item()) + 1 if n > 0 else 0

    # A fresh generator is used for the feature generation.
    g = torch.Generator().manual_seed(seed)

    # Draw one centroid for every block.
    centroids = torch.randn((num_blocks, dim), generator=g)

    # Draw one independent standard-normal noise vector per node
    # using the same generator.
    node_noise = torch.randn((n, dim), generator=g)

    # Each node is centered at the centroid of its block.
    features = centroids[blocks] + noise * node_noise

    return features

# Step 4 - edge_split
def edge_split(edge_index, n, val_frac, test_frac, seed):
    # Get unique undirected edges and sort them for deterministic ordering
    # before applying the seeded shuffle.
    undirected_edges = sorted(undirected_edge_set(edge_index))

    m = len(undirected_edges)

    # Shuffle the edge indices using the specified seed.
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(m, generator=g)

    # Number of validation and test edges.
    m_val = round(val_frac * m)
    m_test = round(test_frac * m)

    # The first shuffled edges go to validation, the next ones to test,
    # and the remaining edges are used for training.
    val_indices = perm[:m_val]
    test_indices = perm[m_val:m_val + m_test]
    train_indices = perm[m_val + m_test:]

    def make_pos_edges(indices):
        # Always return a long tensor of shape (2, number_of_edges),
        # including when the set is empty.
        if indices.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long)

        edges = torch.tensor(
            [undirected_edges[i] for i in indices.tolist()],
            dtype=torch.long,
        )

        return edges.t().contiguous()

    train_pos = make_pos_edges(train_indices)
    val_pos = make_pos_edges(val_indices)
    test_pos = make_pos_edges(test_indices)

    # Build the training graph with both directions of every training edge.
    train_reverse = train_pos[[1, 0], :]
    train_edge_index = torch.cat([train_pos, train_reverse], dim=1)

    return {
        "train_pos": train_pos,
        "train_edge_index": train_edge_index,
        "val_pos": val_pos,
        "test_pos": test_pos,
        "num_nodes": n,
    }

# Step 5 - sample_negative_edges
import random

def sample_negative_edges(edge_index, n, num, seed, exclude=None):
    # Existing graph edges are stored as canonical undirected pairs.
    edge_set = undirected_edge_set(edge_index)

    # Optionally exclude another collection of edges as well.
    exclude_set = undirected_edge_set(exclude) if exclude is not None else set()

    # Use Python's random.Random exactly as specified.
    rng = random.Random(seed)

    sampled = []
    sampled_set = set()

    # Rejection sampling until the requested number of negatives is found.
    while len(sampled) < num:
        u = rng.randrange(n)
        v = rng.randrange(n)

        # Ignore self-loops.
        if u == v:
            continue

        pair = (min(u, v), max(u, v))

        # Reject existing edges, excluded edges, and duplicate samples.
        if pair in edge_set:
            continue
        if pair in exclude_set:
            continue
        if pair in sampled_set:
            continue

        sampled.append(pair)
        sampled_set.add(pair)

    # Return the pairs in the order in which they were sampled.
    if num == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(sampled, dtype=torch.long).t().contiguous()

# Step 6 - uniform_random_walks
def uniform_random_walks(adj, walk_length, walks_per_node, seed):
    rng = random.Random(seed)

    n = len(adj)
    walks = []

    # Generate walks in pass order, then node index order.
    for _ in range(walks_per_node):
        for start_node in range(n):
            walk = [start_node]
            current_node = start_node

            # Extend the walk until it reaches the requested length.
            while len(walk) < walk_length:
                neighbours = adj[current_node]

                if neighbours:
                    current_node = rng.choice(neighbours)
                else:
                    # At a dead end, repeat the current node.
                    current_node = current_node

                walk.append(current_node)

            walks.append(walk)

    return torch.tensor(walks, dtype=torch.long)

# Step 7 - node2vec_walks
def node2vec_transition_weights(adj, prev, cur, p, q):
    weights = []

    # Use a set for efficient membership checks when determining
    # whether a candidate neighbour is also adjacent to `prev`.
    prev_neighbours = set(adj[prev])

    for neighbour in adj[cur]:
        if neighbour == prev:
            # Return to the previous node.
            weights.append(1.0 / p)
        elif neighbour in prev_neighbours:
            # Move to a node connected to the previous node.
            weights.append(1.0)
        else:
            # Move to a node that is not connected to the previous node.
            weights.append(1.0 / q)

    return weights


def node2vec_walks(adj, walk_length, walks_per_node, p, q, seed):
    rng = random.Random(seed)

    n = len(adj)
    walks = []

    # Generate walks in the same order as uniform_random_walks:
    # pass order first, then node index order.
    for _ in range(walks_per_node):
        for start_node in range(n):
            walk = [start_node]
            current_node = start_node
            prev_node = None

            while len(walk) < walk_length:
                neighbours = adj[current_node]

                if not neighbours:
                    # At a dead end, repeat the current node.
                    walk.append(current_node)
                    continue

                if prev_node is None:
                    # The first step is unbiased.
                    next_node = rng.choice(neighbours)
                else:
                    # Later steps use node2vec's biased transition weights.
                    weights = node2vec_transition_weights(
                        adj,
                        prev_node,
                        current_node,
                        p,
                        q,
                    )
                    next_node = rng.choices(
                        neighbours,
                        weights=weights,
                        k=1,
                    )[0]

                prev_node = current_node
                current_node = next_node
                walk.append(current_node)

            walks.append(walk)

    return torch.tensor(walks, dtype=torch.long)

