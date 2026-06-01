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
    graph_path = "../dataset/Art500K/Art500KGraph.pt"
    image_path = "../dataset/Art500K/images/"

    data = torch.load(graph_path)

    text = data.titles
    num_nodes = len(text)
    data.num_nodes = len(text)

    images = []
    for i in range(num_nodes):
        node_image_path = image_path + f"{i+1}.jpg"
        images.append(node_image_path)


    data.images = images

    return data, text


    
