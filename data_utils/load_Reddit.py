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
    file_path = f"../../dataset/Reddit/"
    image_path = f"../../dataset/Reddit/images/"
    csv_path = file_path + 'Reddit.csv'

    g = dgl.load_graphs(file_path + 'RedditGraph.pt')
    g = g[0][0]
    src_nodes, dst_nodes = g.edges()
    edge_index = torch.stack([src_nodes, dst_nodes], dim=0)

    df = pd.read_csv(csv_path)
    label = df['label']
    label = label.to_list()
    num_nodes = len(label)
    y = torch.tensor(label)

    category_label_mapping = dict(zip(df['label'], df['subreddit']))

    text = df['caption']
    text = text.to_list()

    text_label = df['subreddit']
    text_label = text_label.to_list()

    images = []
    for i in range(num_nodes):
        node_image_path = image_path + f"{i}.jpg"
        images.append(node_image_path)
    
    data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes, images=images, category_label_mapping=category_label_mapping, text_label=text_label)
    return data, text