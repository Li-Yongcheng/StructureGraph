#!/usr/bin/env bash

# Test for running a single experiment. --repeat means run how many different random seeds.
# python main.py --cfg configs/pyg/example_node.yaml --repeat 3 # node classification
# python main.py --cfg configs/pyg/example_link.yaml --repeat 3 # link prediction
# python main.py --cfg configs/tudataset_graph_grid_tudataset_graph/tudataset_graph-format=PyG-dataset=TU_ENZYMES-task=graph-trans=False-l_pre=1-l_mp=2-l_post=2-stage=skipconcat-agg=add.yaml --repeat 3 # graph classification
python main.py --cfg configs/hetero_graph.yaml --repeat 4
