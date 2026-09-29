"""
Node Embeddings and Link Prediction from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Node Embeddings and Link Prediction from Scratch.

Story: embed Zachary's karate club with DeepWalk and check that nearest
neighbours share a faction; predict held-out edges of a stochastic block model
with skip-gram embeddings and measure how much the score inflates when the test
edges leak into the walks; train an inductive GraphSAGE encoder and score edges
of nodes that did not exist at training time, against the lookup table that has
nothing for them; and complete a synthetic knowledge graph with TransE, scored
with filtered ranks before and after training.
"""
import torch


def main() -> None:
    torch.manual_seed(0)
    lines = link_prediction_report(sizes=(40, 40, 40), p_in=0.25, p_out=0.02, seed=0, kg_entities=60, kg_relations=8, sage_steps=150, kg_epochs=100)

    print("1. Embeddings from walks")
    print("   " + lines[0])
    print("   The skip-gram objective never saw a label; the geometry alone separates the two factions.")

    print("\n2. Link prediction, honest and leaked")
    print("   " + lines[1])
    print("   Same model, same test edges, same negatives. One changed line and the score is a memory, not a prediction.")

    print("\n3. Inductive versus transductive")
    print("   " + lines[2])
    print("   The encoder computes an embedding from features and neighbours; the table has an untrained row for every new node.")

    print("\n4. Knowledge graph completion")
    print("   " + lines[3])
    print("   Relations as translations: on a graph generated from that assumption, training moves the true answer into the top ten for most held-out facts.")


if __name__ == "__main__":
    main()

