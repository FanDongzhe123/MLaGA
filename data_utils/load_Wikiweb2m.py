import os
import sys
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
import dgl
from PIL import Image
from tqdm import tqdm
import json
import pdb

def build_pygData(full_text=True, if_test=False):
    # graph_path = f"../dataset/VideoGames/{main_category}_graph.dgl"
    graph_path = f"/vast/df2362/wikiweb2m/data/Wikiweb2mGraph.pt"
    img_dir = f"/vast/df2362/wikiweb2m/graph_images"


    graphdata_dict = torch.load(graph_path)
    graphdata = graphdata_dict['WikiWeb2MGraph']
    edge_index = graphdata.edge_index
    text = graphdata.text
    images = graphdata.images
    y = graphdata.y
    text_label = graphdata.text_label

    num_nodes = len(text_label)

    data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes, images=images, text_label=text_label)
    # data.train_id, data.val_id, data.test_id, data.train_mask, data.val_mask, data.test_mask =  split_graph_Ratio(42, num_nodes, train_ratio=0.6, val_ratio=0.2)
    # data.semi_train_id, data.semi_train_mask = split_graph_Ratio_semi(42, data.train_mask, train_ratio=0.5)

    return data, text