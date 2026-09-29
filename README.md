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
- [x] **5.** sample_negative_edges
- [x] **6.** uniform_random_walks
- [x] **7.** node2vec_walks
- [x] **8.** skipgram_pairs
- [x] **9.** SkipGramModel
- [x] **10.** train_skipgram
- [x] **11.** knn_label_agreement
- [x] **12.** dot_decoder
- [x] **13.** roc_auc
- [x] **14.** link_bce_loss
- [x] **15.** evaluate_link_prediction
- [x] **16.** embedding_link_prediction
- [x] **17.** leakage_experiment
- [x] **18.** sample_neighbors
- [x] **19.** SAGEConv
- [x] **20.** GraphSAGE
- [x] **21.** train_sage_link_predictor
- [x] **22.** inductive_evaluation
- [x] **23.** synthetic_kg
- [x] **24.** TransE
- [x] **25.** corrupt_triples
- [x] **26.** train_transe
- [x] **27.** filtered_rank
- [x] **28.** link_prediction_report

## Results

```
1. Embeddings from walks
   karate club: agreement=0.9706 random=0.3529
   The skip-gram objective never saw a label; the geometry alone separates the two factions.

2. Link prediction, honest and leaked
   leakage: honest_auc=0.7646 leaked_auc=0.8526 honest_mrr=0.0513 leaked_mrr=0.0748
   Same model, same test edges, same negatives. One changed line and the score is a memory, not a prediction.

3. Inductive versus transductive
   inductive: sage_auc=0.8174 skipgram_auc=0.5135 new_nodes=24
   The encoder computes an embedding from features and neighbours; the table has an untrained row for every new node.

4. Knowledge graph completion
   TransE: before_mrr=0.0749 before_hits@10=0.1646 after_mrr=0.5245 after_hits@10=0.8924
   Relations as translations: on a graph generated from that assumption, training moves the true answer into the top ten for most held-out facts.
```
