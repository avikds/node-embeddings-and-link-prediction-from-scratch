# Node Embeddings and Link Prediction from Scratch

The half of graph learning that message passing does not cover, built in pure PyTorch on Zachary's karate club and synthetic stochastic block models. Turn a graph into a corpus of random walks, DeepWalk's uniform ones and node2vec's biased ones, and train a skip-gram model with negative sampling until the embedding recovers the club's split. Then predict missing edges: a leakage-safe edge split, dot-product decoders, sampled negatives, AUC, hits@k and MRR, and the experiment that shows how much a score inflates when test edges leak into the walks. Replace lookup embeddings with an inductive GraphSAGE encoder that samples neighbors, and evaluate it on nodes that did not exist at training time. Finish with TransE on a synthetic knowledge graph, trained with corrupted triples and scored with filtered ranks.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** karate_club_graph
- [x] **2.** build_adjacency_lists
- [x] **3.** sbm_graph
- [x] **4.** edge_split

---

Built on Deep-ML.
