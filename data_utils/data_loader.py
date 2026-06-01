import os
import torch
import sys
import sys
sys.path.append("..")
sys.path.append(".")
import numpy as np 
import pandas as pd 
import pdb
from data_utils.eval_utils import split_graph_Ratio

def split_data_pyg(dataset_name,data,seed,split):
    num_nodes = len(data.y)


    if dataset_name in ['Movies', 'Grocery', 'Toys'] and split == 'supervised':
        data.train_id, data.val_id, data.test_id, data.train_mask, data.val_mask, data.test_mask =  split_graph_Ratio(seed, num_nodes, train_ratio=0.6, val_ratio=0.2)
    elif dataset_name in ['Amazon-cloth', 'Amazon-sports']:
        pass
    return data


def load_data(dataset_name, use_text=True, use_feature=False, full_text=True, if_test=False, model_epoch=1): 
    print(f'Loading {dataset_name} dataset')
    if dataset_name == 'Toys':
        from data_utils.load_Toys import build_pygData
    elif dataset_name == 'Movies':
        from data_utils.load_Movies import build_pygData
    elif dataset_name == 'Grocery':
        from data_utils.load_Grocery import build_pygData
    elif dataset_name == 'Ele-fashion':
        from data_utils.load_ELe_fashion import build_pygData
    elif dataset_name == 'Goodreads':
        from data_utils.load_Goodreads import build_pygData
    elif dataset_name == 'Amazon-cloth':
        from data_utils.load_Amazon_cloth import build_pygData
    elif dataset_name == 'Amazon-sports':
        from data_utils.load_Amazon_sports import build_pygData
    elif dataset_name == 'Reddit':
        from data_utils.load_Reddit import build_pygData
    elif dataset_name == 'Movies_new':
        from data_utils.load_Movies_new import build_pygData
    elif dataset_name == 'Health':
        from data_utils.load_Health import build_pygData
    elif dataset_name == 'VideoGames':
        from data_utils.load_VideoGames import build_pygData
    elif dataset_name == 'Beauty':
        from data_utils.load_Beauty import build_pygData
    elif dataset_name == 'Arts':
        from data_utils.load_Arts import build_pygData
    elif dataset_name == 'Automotive':
        from data_utils.load_Automotive import build_pygData
    elif dataset_name == 'CD':
        from data_utils.load_CD import build_pygData
    elif dataset_name == 'RedditS':
        from data_utils.load_RedditS import build_pygData
    elif dataset_name == 'Art500K':
        from data_utils.load_Art500K import build_pygData
    elif dataset_name == 'Wikiweb2m':
        from data_utils.load_Wikiweb2m import build_pygData
    
    data, text = build_pygData(full_text, if_test)
    # if dataset_name in ['Toys', 'Grocery', 'Movies', 'Reddit']:
    #     data = split_data_pyg(dataset_name, data, seed, split)
    if use_text == False:
        return data
    
    if use_feature:
        # emb_path = f'../Model/Embeddings/{dataset_name}/{dataset_name}.pt'
        print(f'Loading {dataset_name} features of epoch {model_epoch}')
        if dataset_name == "Goodreads":
            emb_path = '/vast/df2362/Goodreads/query_token_all.pt'
        else:
            emb_path = f'../../dataset/{dataset_name}/query_token_dino_sbert_dino_sbert.pt'

        features = torch.load(emb_path)
        features = torch.mean(features, dim=1)
    else:
        features = None


    data.x = features

    return data, text

if __name__ == "__main__":
    datasets = ["Movies", "Toys", "VideoGames", "Arts"]
    for dataset in datasets:
        data, text = load_data(dataset, use_feature=True)
        torch.save(data.x, f'../dataset/{dataset}/clip_feat.pt')

