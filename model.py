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

# Step 8 - skipgram_pairs
def skipgram_pairs(walks, window):
    pairs = []

    # Process walks in their existing order.
    for walk in walks.tolist():
        walk_length = len(walk)

        # For each position, generate context positions in increasing
        # position order within the specified window.
        for i in range(walk_length):
            center = walk[i]

            start = max(0, i - window)
            end = min(walk_length, i + window + 1)

            for j in range(start, end):
                # Exclude the center position itself and contexts whose
                # node is identical to the center node.
                if j == i or walk[j] == center:
                    continue

                pairs.append((center, walk[j]))

    # Always return a long tensor with shape (2, P), including (2, 0).
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def negative_sampling_distribution(walks, n, power=0.75):
    # Count how many times each node appears across all walks.
    counts = torch.bincount(
        walks.reshape(-1),
        minlength=n,
    ).to(dtype=torch.float32)

    # Raise visit counts to the requested power.
    weights = counts.pow(power)

    # Normalize into a probability distribution.
    total = weights.sum()

    if total == 0:
        return torch.zeros(n, dtype=torch.float32)

    return weights / total


def sample_negatives(probs, num_pairs, k, generator):
    # Draw k negative node IDs independently for every positive pair.
    return torch.multinomial(
        probs,
        num_pairs * k,
        replacement=True,
        generator=generator,
    ).reshape(num_pairs, k)

# Step 9 - SkipGramModel
class SkipGramModel(nn.Module):
    def __init__(self, n, dim, seed=0):
        super().__init__()

        # Make parameter initialization deterministic under the given seed.
        torch.manual_seed(seed)

        self.in_embed = nn.Embedding(n, dim)
        self.out_embed = nn.Embedding(n, dim)

        # Word2Vec-style initialization for the input embeddings.
        with torch.no_grad():
            self.in_embed.weight.uniform_(
                -0.5 / dim,
                0.5 / dim
            )

            # Output embeddings start at zero.
            self.out_embed.weight.zero_()

    def forward(self, center, context, negatives):
        # Input embeddings for the center nodes: (B, D)
        v_center = self.in_embed(center)

        # Output embeddings for the positive context nodes: (B, D)
        u_context = self.out_embed(context)

        # Positive scores: u_o^T v_c -> (B,)
        positive_scores = (u_context * v_center).sum(dim=1)

        # Output embeddings for negative samples: (B, K, D)
        u_negative = self.out_embed(negatives)

        # Negative scores: u_k^T v_c -> (B, K)
        negative_scores = (
            u_negative * v_center.unsqueeze(1)
        ).sum(dim=2)

        # Skip-gram negative-sampling objective:
        # -log(sigmoid(u_o^T v_c))
        # -sum_k log(sigmoid(-u_k^T v_c))
        positive_loss = -F.logsigmoid(positive_scores)
        negative_loss = -F.logsigmoid(-negative_scores).sum(dim=1)

        # Mean loss over the batch.
        return (positive_loss + negative_loss).mean()

    def embeddings(self):
        # Return detached copies of the input embeddings used by the project.
        return self.in_embed.weight.detach().clone()

# Step 10 - train_skipgram
def train_skipgram(model, pairs, neg_probs, k, epochs, batch_size, lr, seed):
    # Use one generator for both pair shuffling and negative sampling.
    g = torch.Generator().manual_seed(seed)

    # Adam optimizer with the requested learning rate.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_pairs = pairs.shape[1]
    epoch_losses = []

    model.train()

    for _ in range(epochs):
        # Shuffle the pair columns using the seeded generator.
        perm = torch.randperm(num_pairs, generator=g)

        total_loss = 0.0
        total_examples = 0

        # Process shuffled pairs in consecutive batches.
        for start in range(0, num_pairs, batch_size):
            batch_indices = perm[start:start + batch_size]
            batch_size_actual = batch_indices.numel()

            batch = pairs[:, batch_indices]
            center = batch[0]
            context = batch[1]

            # Sample k negatives for every positive pair.
            negatives = sample_negatives(
                neg_probs,
                batch_size_actual,
                k,
                g,
            )

            optimizer.zero_grad()

            loss = model(center, context, negatives)

            loss.backward()
            optimizer.step()

            # Accumulate a batch-size-weighted loss so that the
            # epoch mean is computed over individual training pairs.
            total_loss += float(loss.detach()) * batch_size_actual
            total_examples += batch_size_actual

        # Guard against an empty pair set.
        if total_examples == 0:
            epoch_losses.append(0.0)
        else:
            epoch_losses.append(total_loss / total_examples)

    return epoch_losses

# Step 11 - knn_label_agreement
def knn_label_agreement(Z, labels, k):
    # Normalize every embedding vector so that the dot product
    # becomes cosine similarity.
    Z_normalized = F.normalize(Z, p=2, dim=1)

    # Compute the full pairwise cosine-similarity matrix.
    similarities = Z_normalized @ Z_normalized.t()

    # A node must not be included in its own neighbourhood.
    similarities.fill_diagonal_(-float("inf"))

    # Select the k most similar nodes for every node.
    neighbours = similarities.topk(k, dim=1).indices

    # Get the labels of the k nearest neighbours.
    neighbour_labels = labels[neighbours]

    # Majority vote among the k neighbours.
    majority_labels = torch.mode(neighbour_labels, dim=1).values

    # Fraction of nodes whose predicted majority label matches
    # their own ground-truth label.
    return float((majority_labels == labels).float().mean())


def embed_karate_club(
    dim=16,
    walk_length=10,
    walks_per_node=20,
    window=3,
    k=5,
    epochs=5,
    seed=0,
):
    # Load the Karate Club graph and its faction labels.
    edge_index, labels = karate_club_graph()

    # Build the undirected adjacency lists required for random walks.
    adj = build_adjacency_lists(edge_index, 34)

    # Generate the DeepWalk corpus using uniform random walks.
    walks = uniform_random_walks(
        adj,
        walk_length,
        walks_per_node,
        seed,
    )

    # Convert walks into center-context skip-gram pairs.
    pairs = skipgram_pairs(walks, window)

    # Build the 0.75-power negative-sampling distribution.
    neg_probs = negative_sampling_distribution(
        walks,
        34,
        power=0.75,
    )

    # Create the skip-gram model using the requested seed.
    model = SkipGramModel(34, dim, seed=seed)

    # Train with 5 negative samples per positive pair,
    # batch size 256, learning rate 0.01, and the requested seed.
    losses = train_skipgram(
        model,
        pairs,
        neg_probs,
        k=5,
        epochs=epochs,
        batch_size=256,
        lr=0.01,
        seed=seed,
    )

    # The project uses the input embedding vectors as node embeddings.
    Z = model.embeddings()

    return Z, labels, losses

# Step 12 - dot_decoder
def dot_decoder(Z, pairs):
    # Select the embedding vectors for the two endpoints of each pair.
    z_u = Z[pairs[0]]
    z_v = Z[pairs[1]]

    # Compute one dot product per pair.
    return (z_u * z_v).sum(dim=1)


def cosine_decoder(Z, pairs):
    # Normalize the node embeddings so their dot products are
    # cosine similarities.
    Z_normalized = F.normalize(Z, p=2, dim=1)

    z_u = Z_normalized[pairs[0]]
    z_v = Z_normalized[pairs[1]]

    return (z_u * z_v).sum(dim=1)


def hadamard_features(Z, pairs):
    # Elementwise product of the two endpoint embeddings.
    z_u = Z[pairs[0]]
    z_v = Z[pairs[1]]

    return z_u * z_v

# Step 13 - roc_auc
def roc_auc(pos, neg):
    # Compare every positive score against every negative score.
    comparisons = pos.unsqueeze(1) - neg.unsqueeze(0)

    # A positive strictly above a negative contributes 1.
    # A tie contributes 0.5.
    scores = (
        (comparisons > 0).float()
        + 0.5 * (comparisons == 0).float()
    )

    return float(scores.mean())


def hits_at_k(pos, neg, k):
    # For each positive, rank is one plus the number of negatives
    # with a strictly higher score.
    ranks = 1 + (neg.unsqueeze(0) > pos.unsqueeze(1)).sum(dim=1)

    # Fraction of positives whose rank is at most k.
    return float((ranks <= k).float().mean())


def mean_reciprocal_rank(pos, neg):
    # Compute the rank of every positive against all negatives.
    ranks = 1 + (neg.unsqueeze(0) > pos.unsqueeze(1)).sum(dim=1)

    # Mean reciprocal rank.
    return float((1.0 / ranks.float()).mean())

# Step 14 - link_bce_loss
def link_bce_loss(pos, neg):
    # Positive edges are targets of 1.
    pos_targets = torch.ones_like(pos)

    # Negative edges are targets of 0.
    neg_targets = torch.zeros_like(neg)

    # Compute BCE-with-logits separately for positive and negative
    # scores, then add the two mean losses.
    pos_loss = F.binary_cross_entropy_with_logits(
        pos,
        pos_targets,
    )

    neg_loss = F.binary_cross_entropy_with_logits(
        neg,
        neg_targets,
    )

    return pos_loss + neg_loss


def link_margin_loss(pos, neg, margin):
    # Compare every positive score against every negative score.
    # Each pair contributes max(0, margin - pos + neg).
    losses = torch.relu(
        margin - pos.unsqueeze(1) + neg.unsqueeze(0)
    )

    return losses.mean()

# Step 15 - evaluate_link_prediction
def evaluate_link_prediction(score_fn, split, num_neg, seed):
    # Combine validation and test positives so neither can ever be
    # sampled as a negative edge.
    excluded = torch.cat(
        [split["val_pos"], split["test_pos"]],
        dim=1,
    )

    # Sample negative edges only from non-edges of the training graph,
    # while also excluding validation and test positives.
    neg_pairs = sample_negative_edges(
        split["train_edge_index"],
        split["num_nodes"],
        num_neg,
        seed=seed,
        exclude=excluded,
    )

    # Score the held-out test positives and sampled negatives.
    pos_scores = score_fn(split["test_pos"])
    neg_scores = score_fn(neg_pairs)

    # Evaluate the ranking quality of the predictions.
    return {
        "auc": roc_auc(pos_scores, neg_scores),
        "hits@10": hits_at_k(pos_scores, neg_scores, 10),
        "mrr": mean_reciprocal_rank(pos_scores, neg_scores),
    }

# Step 16 - embedding_link_prediction
def embedding_link_prediction(
    split,
    dim=32,
    walk_length=15,
    walks_per_node=8,
    window=4,
    k=5,
    epochs=3,
    seed=0,
    num_neg=500,
):
    num_nodes = split["num_nodes"]

    # Build adjacency lists using only the training graph.
    adj = build_adjacency_lists(
        split["train_edge_index"],
        num_nodes,
    )

    # Generate DeepWalk-style uniform random walks from the
    # training graph only.
    walks = uniform_random_walks(
        adj,
        walk_length,
        walks_per_node,
        seed=seed,
    )

    # Convert walks into skip-gram center-context pairs.
    pairs = skipgram_pairs(
        walks,
        window,
    )

    # Build the negative-sampling distribution from the walk corpus.
    neg_probs = negative_sampling_distribution(
        walks,
        num_nodes,
        power=0.75,
    )

    # Train the skip-gram node embedding model.
    model = SkipGramModel(
        num_nodes,
        dim,
        seed=seed,
    )

    train_skipgram(
        model,
        pairs,
        neg_probs,
        k=k,
        epochs=epochs,
        batch_size=512,
        lr=0.01,
        seed=seed,
    )

    # The trained input embeddings are the node representations
    # used by the link-prediction decoder.
    Z = model.embeddings()

    # Use dot-product scores to evaluate the held-out test edges.
    metrics = evaluate_link_prediction(
        lambda pairs: dot_decoder(Z, pairs),
        split,
        num_neg=num_neg,
        seed=seed,
    )

    return Z, metrics

# Step 17 - leakage_experiment
def leakage_experiment(sizes, p_in, p_out, seed, **kwargs):
    # Build the synthetic stochastic block model.
    edge_index, blocks = sbm_graph(
        sizes,
        p_in,
        p_out,
        seed=seed,
    )

    n = sum(sizes)

    # Create the leakage-safe train/validation/test split.
    split = edge_split(
        edge_index,
        n,
        0.1,
        0.2,
        seed=seed,
    )

    # Honest experiment: the embedding pipeline sees only the
    # training graph when generating random walks.
    _, honest_metrics = embedding_link_prediction(
        split,
        seed=seed,
        **kwargs,
    )

    # Create a separate split dictionary for the leakage experiment.
    leaked_split = dict(split)

    # Deliberately expose the full graph to the embedding pipeline,
    # including validation and test edges. Evaluation remains unchanged.
    leaked_split["train_edge_index"] = edge_index

    # Leaked experiment: random walks can now traverse held-out edges.
    _, leaked_metrics = embedding_link_prediction(
        leaked_split,
        seed=seed,
        **kwargs,
    )

    return {
        "honest_auc": honest_metrics["auc"],
        "leaked_auc": leaked_metrics["auc"],
        "honest_mrr": honest_metrics["mrr"],
        "leaked_mrr": leaked_metrics["mrr"],
    }

# Step 18 - sample_neighbors
def sample_neighbors(adj, nodes, num_samples, rng):
    samples = []

    # Process each requested node independently.
    for node in nodes.tolist():
        neighbours = adj[node]

        if neighbours:
            # Sample with replacement by repeatedly using rng.choice.
            node_samples = [
                rng.choice(neighbours)
                for _ in range(num_samples)
            ]
        else:
            # Isolated nodes repeat themselves.
            node_samples = [node] * num_samples

        samples.append(node_samples)

    # Preserve the required shape even when nodes or num_samples is zero.
    return torch.tensor(
        samples,
        dtype=torch.long,
    ).reshape(len(nodes), num_samples)

# Step 19 - SAGEConv
class SAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        # Linear transformation for the target node's own features.
        self.lin_self = nn.Linear(in_dim, out_dim)

        # Linear transformation for the aggregated neighbour features.
        # The neighbour transformation has no bias.
        self.lin_neigh = nn.Linear(
            in_dim,
            out_dim,
            bias=False,
        )

    def forward(self, x_self, x_neigh, activate=True):
        # Mean aggregation over the sampled neighbours.
        neigh_mean = x_neigh.mean(dim=1)

        # Combine transformed self features and neighbour features.
        h = self.lin_self(x_self) + self.lin_neigh(neigh_mean)

        # Apply ReLU when requested.
        if activate:
            h = F.relu(h)

        # Normalize each output row to unit L2 norm.
        return F.normalize(h, p=2, dim=1)

# Step 20 - GraphSAGE
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, seed=0):
        super().__init__()

        # Make parameter initialization deterministic.
        torch.manual_seed(seed)

        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, adj, nodes, num_samples, rng):
        s1, s2 = num_samples
        batch_size = nodes.numel()

        # Sample first-hop neighbours for every target node.
        # Shape: (B, s1)
        hop1_nodes = sample_neighbors(
            adj,
            nodes,
            s1,
            rng,
        )

        # Flatten first-hop nodes so that each sampled neighbour
        # can independently receive its own second-hop samples.
        hop1_flat = hop1_nodes.reshape(-1)

        # Sample second-hop neighbours for every first-hop node.
        # Shape: (B * s1, s2)
        hop2_nodes = sample_neighbors(
            adj,
            hop1_flat,
            s2,
            rng,
        )

        # First GraphSAGE layer for the target nodes.
        x_self = x[nodes]
        x_neigh = x[hop1_nodes]

        h_target = self.conv1(
            x_self,
            x_neigh,
        )

        # First GraphSAGE layer for every first-hop node.
        x_hop1_self = x[hop1_flat]
        x_hop1_neigh = x[hop2_nodes]

        h_hop1 = self.conv1(
            x_hop1_self,
            x_hop1_neigh,
        )

        # Restore the (B, s1, hidden_dim) structure of the
        # first-hop representations.
        h_hop1 = h_hop1.reshape(
            batch_size,
            s1,
            -1,
        )

        # Second GraphSAGE layer for the target nodes.
        # The second layer has no activation.
        h_target = self.conv2(
            h_target,
            h_hop1,
            activate=False,
        )

        return h_target


def sage_embed_all(model, x, adj, num_samples, rng, batch_size=256):
    n = x.shape[0]
    embeddings = []

    # Evaluation does not require gradients.
    with torch.no_grad():
        for start in range(0, n, batch_size):
            nodes = torch.arange(
                start,
                min(start + batch_size, n),
                dtype=torch.long,
            )

            batch_embeddings = model(
                x,
                adj,
                nodes,
                num_samples,
                rng,
            )

            embeddings.append(batch_embeddings)

    # Concatenate all node batches into one (n, out_dim) matrix.
    if not embeddings:
        return torch.empty(
            (0, model.conv2.lin_self.out_features),
            dtype=x.dtype,
            device=x.device,
        )

    return torch.cat(embeddings, dim=0)

