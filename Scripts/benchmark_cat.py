"""
Benchmark the speed of CAT (+ spiderweb)
"""

import time

import numpy as np
from torch_geometric.data import Data
import torch_geometric
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset
from ogb.graphproppred import PygGraphPropPredDataset
import pandas as pd

from Misc.config import config 
from Misc.utils import edge_tensor_to_list
from Misc.cyclic_adjacency_transform import CyclicAdjacencyTransform

def main():
    repeats = 10
    results = []
    for ds_name in ["molhiv", "molbace", "molbbbp", "molsider", "moltoxcast", "molesol", "mollipo", "moltox21"]:
        for use_spider_web in [False]:
            print(f"\nRunning on {ds_name} \t Spiderweb: {use_spider_web}")

            if ds_name == "zinc":
                ds = ZINC(root=config.DATA_PATH, subset=True, split="train")
            else:
                ds = PygGraphPropPredDataset(root=config.DATA_PATH, name="ogbg-"+ds_name)
                split_idx = ds.get_idx_split()
                ds = ds[split_idx["train"]]
            
            transform = CyclicAdjacencyTransform(debug=True, spiderweb=use_spider_web)
            runtimes = []
                
            # Add one iteration for warmup
            for r in range(repeats + 1):
                transformed_data = []
                start = time.time()
                for data in ds:        
                    transformed_data.append(transform(data))

                end = time.time()
                print(f"\tRuntime  {end - start:.2f}s on {ds_name}")
                runtimes.append(end - start)
                
            # Remove warmup iteration
            runtimes = runtimes[1:]
            print(f"{ds_name}\nRuntime: {np.mean(runtimes):.2f} +/- {np.std(runtimes):.2f}")
            results.append(f"{ds_name} {'+Spider' if use_spider_web else ''}:{np.mean(runtimes):.2f} +/- {np.std(runtimes):.2f}")
            
    print("\n\n\n", "Results\n", "-"*20)
    print("\n\n".join(results))
    
if __name__ == "__main__":
    main()