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
    if if_test:
        image_path = f"../../dataset/Movies/images/"
        file_path = f"../../dataset/Movies/"
        csv_path = file_path + 'Movies.csv'
        split_path = file_path + f'Movies_split.json'
    else:
        image_path = f"../dataset/Movies/images/"
        file_path = f"../dataset/Movies/"
        csv_path = file_path + 'Movies.csv'
        split_path = file_path + f'Movies_split.json'
    
    # csv_path = file_path + 'Movies.csv'

    g = dgl.load_graphs(file_path + 'MoviesGraph.pt')
    g = g[0][0]
    src_nodes, dst_nodes = g.edges()
    edge_index = torch.stack([src_nodes, dst_nodes], dim=0)

    df = pd.read_csv(csv_path)
    label = df['label']
    label = label.to_list()
    text_label = df['second_category']
    text_label = text_label.to_list()
    num_nodes = len(label)
    y = torch.tensor(label)

    category_label_mapping = dict(zip(df['label'], df['second_category']))
    
    if full_text:
        text = df['text']
        text = text.to_list()
    else:
        text = df['title']
        text = text.to_list()

    with open(split_path, 'r') as file:
        split = json.load(file)
    train_indices = split['train']
    val_indices = split['val']
    test_indices = split['test']

    train_indices.sort()
    val_indices.sort()
    test_indices.sort()

    train_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)

    train_mask[train_indices] = True
    test_mask[test_indices] = True
    val_mask[val_indices] = True

    images = []
    for i in range(num_nodes):
        node_image_path = image_path + f"{i}.jpg"
        images.append(node_image_path)

    data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes, images=images, category_label_mapping=category_label_mapping, text_label=text_label, train_id=train_indices, val_id=val_indices, test_id=test_indices, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return data, text


