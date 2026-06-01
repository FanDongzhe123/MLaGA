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
from data_utils.eval_utils import split_graph_Ratio

def build_pygData(full_text=True, if_test=False):
    main_category = "9_CDs_and_Vinyl"
    # graph_path = f"../dataset/VideoGames/{main_category}_graph.dgl"
    graph_path = f"../dataset/CD/CDGraph.pt"
    img_dir = f"../dataset/amazon_images_new/{main_category}"


    graph = torch.load(graph_path, weights_only=True)
    edge_index = graph["adjacency_matrix"].t()

    id_to_label = graph["subcategory"]

    text = []
    label = []
    text_label = []
    images = []
    for node_idx, node in graph["detail"].items():
        asin, desc, title, class_idx, reviews = node.values()
        node_text = f"Title: {title}; Description: {desc}"
        text.append(node_text)
        label.append(class_idx)
        text_label.append(id_to_label[class_idx])

        node_image_path = f"{img_dir}/{asin}.jpg"
        images.append(node_image_path)

    y = torch.tensor(label)
    num_nodes = len(label)

    data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes, images=images, category_label_mapping=id_to_label, text_label=text_label)
    data.train_id, data.val_id, data.test_id, data.train_mask, data.val_mask, data.test_mask =  split_graph_Ratio(42, num_nodes, train_ratio=0.6, val_ratio=0.2)

    return data, text