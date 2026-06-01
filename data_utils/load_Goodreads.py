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

def build_pygData(full_text=True, if_test=False):
    image_path = f"../dataset/Goodreads/images/"
    file_path = f"../dataset/Goodreads/"


    node_mapping_dict = torch.load(file_path + "node_mapping.pt")
    node_mapping_dict = {value: key for key, value in node_mapping_dict.items()}
    images = []
    for key, value in node_mapping_dict.items():
        node_image_path = image_path + f"{value}.jpg"
        images.append(node_image_path)





    label_list = torch.load(file_path + "labels-w-missing.pt")
    y = torch.tensor(label_list)

    edge_list = torch.load(file_path + "nc_edges-nodeid.pt")
    edges_tensor = torch.tensor(edge_list, dtype=torch.long)
    edge_index = edges_tensor.t().contiguous()

    text = []
    with open(file_path + 'books-nc-raw-text.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            text.append(data["raw_text"][0])

    split = torch.load(file_path + 'split.pt')
    train_idx = split['train_idx']
    test_idx = split['test_idx']
    val_idx = split['val_idx']

    num_nodes = len(label_list)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    test_mask[test_idx] = True
    val_mask[val_idx] = True






    data = Data(edge_index=edge_index, y=y, images=images, num_nodes=num_nodes, train_mask=train_mask, test_mask=test_mask, val_mask=val_mask)
    return data, text






