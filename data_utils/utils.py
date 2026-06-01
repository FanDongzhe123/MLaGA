from tqdm import tqdm
from collections import defaultdict
import random
from torch.utils.data import Dataset
import transformers
from transformers import CLIPModel, CLIPImageProcessor
from PIL import Image
import os

#Dataset for MMLLM Stage1
class QformerDataset(Dataset):
    def __init__(self, neighbor_list):
        super(QformerDataset, self).__init__()
        self.neighbor_list = neighbor_list
        # self.image_processor = image_processor
        # self.text_processor = text_processor

    def __len__(self):
        return len(self.neighbor_list)
    
    def __getitem__(self, idx):
        # image_path = self.data.images[index]
        # image = Image.open(image_path)
        item = {}
        item['node_id'] = idx
        item['neighbors'] = self.neighbor_list[idx]
        return item
    
class CLIPDataset(Dataset):
    def __init__(self, image_path, text, processor):
        super(CLIPDataset, self).__init__()
        self.image_path = image_path
        self.text = text
        self.processor = processor
        # self.image_processor = image_processor
        # self.text_processor = text_processor

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        # image_path = self.data.images[index]
        # image = Image.open(image_path)
        image_inputs = self.processor(images=Image.open(self.image_path[idx]), return_tensors="pt")
        image_inputs = image_inputs['pixel_values']

        text_inputs = self.processor(text=self.text[idx], return_tensors="pt", padding="max_length", truncation=True)

        return image_inputs, text_inputs
    
class QformerDataCollator:
    def __call__(self, examples):
        neighbors = [example['neighbors'] for example in examples]
        node_ids = [example['node_id'] for example in examples]

        return{
            "neighbors": neighbors,
            "node_ids": node_ids
        }





# sample neighbors
def build_adjacency_list(edge_index):
    adjacency_list = defaultdict(set)
    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if src != dst:  # Exclude self-loops
            adjacency_list[src].add(dst)
            adjacency_list[dst].add(src)
    return adjacency_list

def collect_neighbors(node, adjacency_list, max_depth):
    current_level = {node}
    visited = set(current_level)
    all_neighbors = []

    for _ in range(max_depth):
        next_level = set()
        for n in current_level:
            for neighbor in adjacency_list.get(n, []):
                if neighbor not in visited:
                    next_level.add(neighbor)
                    visited.add(neighbor)
                    all_neighbors.append(neighbor)
        current_level = next_level
    return all_neighbors

def neighbor_sampler(data):
    tuples_list = []
    edge_index = data.edge_index
    num_nodes = data.num_nodes  
    adjacency_list = build_adjacency_list(edge_index)
    for node_index in tqdm(range(num_nodes)):
        if node_index not in adjacency_list:  
                tuples_list.append([node_index])# Skip isolated nodes
                continue
        all_neighbors = collect_neighbors(node_index, adjacency_list, 1)
        tuples_list.append(all_neighbors)

    return tuples_list



def mkdir_p(path, log=True):

    import errno
    if os.path.exists(path):
        return
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file