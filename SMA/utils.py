import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
from tqdm import tqdm
from collections import defaultdict

def neighbor_sampler(data, samples_size = 5):
    """
    Sample neighbors for each node in the graph.
    
    Args:
    - data: data object that contains the edge index
    - samples_size: number of neighbors to sample for each node
    
    Returns:
    - neighbor_list: list where each element is the list of sampled neighbors for a node
    """
    edge_index = data.edge_index
    num_nodes = max(edge_index[0].max().item(), edge_index[1].max().item()) + 1
    
    # Build the neighbor list for each node
    neighbors = defaultdict(list)
    for src, dst in zip(edge_index[0], edge_index[1]):
        neighbors[src.item()].append(dst.item())
    
    # Sample a fixed number of neighbors for each node
    neighbor_list = []
    for node_id in range(num_nodes):
        if node_id in neighbors and len(neighbors[node_id]) > 0:
            # When there are not enough neighbors, sample with replacement
            if len(neighbors[node_id]) < samples_size:
                sampled = random.choices(neighbors[node_id], k=samples_size)
            else:
                sampled = random.sample(neighbors[node_id], samples_size)
        else:
            # When there are no neighbors, pad with the node itself
            sampled = [node_id] * samples_size
        
        neighbor_list.append(sampled)
    
    return neighbor_list

class QformerDataset(Dataset):
    def __init__(self, neighbor_list, image_paths, texts, processor, vision_encoder_type="clip", text_encoder_type="clip", global_ids=None):
        super(QformerDataset, self).__init__()
        self.neighbor_list = neighbor_list
        self.texts = texts
        self.image_paths = image_paths
        self.processor = processor
        self.vision_encoder_type = vision_encoder_type
        self.text_encoder_type = text_encoder_type
        self.global_ids = global_ids
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Get info for the current node
        node_id = idx
        node_text = self.texts[node_id]
        node_image_path = self.image_paths[node_id]
        
        # Get info for neighbor nodes
        neighbor_ids = self.neighbor_list[node_id]
        neighbor_texts = [self.texts[i] for i in neighbor_ids]
        neighbor_image_paths = [self.image_paths[i] for i in neighbor_ids]
        
        # Process images and texts for the current node and its neighbors
        node_image_inputs, node_text_inputs = self.preprocess_node(node_image_path, node_text)
        neighbor_image_inputs, neighbor_text_inputs = self.preprocess_neighbors(neighbor_image_paths, neighbor_texts)
        
        # Build the return dict
        return_dict = {
            "node_image_inputs": node_image_inputs,
            "node_text_inputs": node_text_inputs,
            "neighbor_image_inputs": neighbor_image_inputs,
            "neighbor_text_inputs": neighbor_text_inputs,
            "node_id": node_id
        }
        
        # Add the global id to the return dict when one is available
        if self.global_ids:
            return_dict["global_id"] = self.global_ids[node_id]
        
        return return_dict
    
    def preprocess_neighbors(self, neighbor_image_paths, neighbor_texts):
        """Process images and texts for neighbor nodes."""
        images = []
        for image_path in neighbor_image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                W, H = image.size
                if H == 1 or W == 1:
                    image = Image.new("RGB", (224, 224), (255, 255, 255))
                images.append(image)
            except Exception as e:
                # If loading the image fails, fall back to a blank image
                print(f"Error loading image {image_path}: {e}")
                images.append(Image.new("RGB", (224, 224), (255, 255, 255)))
        
        # Image processing
        if hasattr(self.processor, 'vision_processor'):
            # Use the custom processor (e.g. DINO)
            image_inputs = self.processor(images=images, return_tensors="pt")
        else:
            # Use the standard CLIP processor
            image_inputs = self.processor(images=images, return_tensors="pt")
        
        # Text processing
        if hasattr(self.processor, 'text_processor'):
            # Use the custom processor
            if self.text_encoder_type == 'sbert':
                # SBERT typically uses max_length=128
                text_inputs = self.processor.text_processor(
                    text=neighbor_texts, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=128, 
                    return_tensors="pt"
                )
            else:
                # CLIP uses max_length=77
                text_inputs = self.processor.text_processor(
                    text=neighbor_texts, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=77, 
                    return_tensors="pt"
                )
        else:
            # Use the standard CLIP processor
            text_inputs = self.processor(
                text=neighbor_texts, 
                padding='max_length', 
                truncation=True, 
                max_length=77, 
                return_tensors="pt"
            )
        
        return image_inputs['pixel_values'], text_inputs
        
    def preprocess_node(self, node_image_path, node_text):
        """Process image and text for the current node."""
        try:
            image = Image.open(node_image_path).convert("RGB")
            W, H = image.size
            if H == 1 or W == 1:
                image = Image.new("RGB", (224, 224), (255, 255, 255))
        except Exception as e:
            # If loading the image fails, fall back to a blank image
            print(f"Error loading image {node_image_path}: {e}")
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        
        # Image processing
        if hasattr(self.processor, 'vision_processor'):
            # Use the custom processor (e.g. DINO)
            image_inputs = self.processor(images=[image], return_tensors="pt")
        else:
            # Use the standard CLIP processor
            image_inputs = self.processor(images=[image], return_tensors="pt")
        
        # Text processing
        if hasattr(self.processor, 'text_processor'):
            # Use the custom processor
            if self.text_encoder_type == 'sbert':
                # SBERT typically uses max_length=128
                text_inputs = self.processor.text_processor(
                    text=[node_text], 
                    padding='max_length', 
                    truncation=True, 
                    max_length=128, 
                    return_tensors="pt"
                )
            else:
                # CLIP uses max_length=77
                text_inputs = self.processor.text_processor(
                    text=[node_text], 
                    padding='max_length', 
                    truncation=True, 
                    max_length=77, 
                    return_tensors="pt"
                )
        else:
            # Use the standard CLIP processor
            text_inputs = self.processor(
                text=[node_text], 
                padding='max_length', 
                truncation=True, 
                max_length=77, 
                return_tensors="pt"
            )
        
        return image_inputs['pixel_values'], text_inputs


class QformerDataCollator:
    def __call__(self, examples):
        node_image_inputs = torch.cat([example["node_image_inputs"] for example in examples])
        
        # Node text inputs
        node_text_input_ids = torch.cat([example["node_text_inputs"]['input_ids'] for example in examples])
        node_text_attention_mask = torch.cat([example["node_text_inputs"]['attention_mask'] for example in examples])
        node_text_inputs = {
            'input_ids': node_text_input_ids,
            'attention_mask': node_text_attention_mask
        }
        
        # Neighbor image inputs
        num_neighbors = examples[0]["neighbor_image_inputs"].size(0)
        neighbor_image_inputs = torch.stack([example["neighbor_image_inputs"] for example in examples])
        
        # Neighbor text inputs
        neighbor_text_input_ids = torch.stack([example["neighbor_text_inputs"]['input_ids'] for example in examples])
        neighbor_text_attention_mask = torch.stack([example["neighbor_text_inputs"]['attention_mask'] for example in examples])
        neighbor_text_inputs = {
            'input_ids': neighbor_text_input_ids,
            'attention_mask': neighbor_text_attention_mask
        }
        
        # Collect node ids and global ids
        node_ids = [example["node_id"] for example in examples]
        
        # Build the return dict
        batch = {
            "node_image_inputs": node_image_inputs,
            "node_text_inputs": node_text_inputs,
            "neighbor_image_inputs": neighbor_image_inputs,
            "neighbor_text_inputs": neighbor_text_inputs,
            "node_ids": node_ids
        }
        
        # Add global ids to the batch when available
        if "global_id" in examples[0]:
            batch["global_ids"] = [example["global_id"] for example in examples]
        
        return batch

# (Add this code to the existing utils_dino.py file)

class QformerEvalDataset(Dataset):
    def __init__(self, image_paths, texts, processor, vision_encoder_type="clip", text_encoder_type="clip"):
        super(QformerEvalDataset, self).__init__()
        self.texts = texts
        self.image_paths = image_paths
        self.processor = processor
        self.vision_encoder_type = vision_encoder_type
        self.text_encoder_type = text_encoder_type
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Get info for the current node
        node_text = self.texts[idx]
        node_image_path = self.image_paths[idx]
        
        # Process image and text for the current node
        node_image_inputs, node_text_inputs = self.preprocess_node(node_image_path, node_text)
        
        # Build the return dict
        return {
            "node_image_inputs": node_image_inputs,
            "node_text_inputs": node_text_inputs,
        }
    
    def preprocess_node(self, node_image_path, node_text):
        """Process image and text for the current node."""
        try:
            image = Image.open(node_image_path).convert("RGB")
            W, H = image.size
            if H == 1 or W == 1:
                image = Image.new("RGB", (224, 224), (255, 255, 255))
        except Exception as e:
            print(f"Error loading image {node_image_path}: {e}")
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        
        # Image processing
        if hasattr(self.processor, 'vision_processor'):
            # Use the custom processor (e.g. DINO)
            image_inputs = self.processor(images=[image], return_tensors="pt")
        else:
            # Use the standard CLIP processor
            image_inputs = self.processor(images=[image], return_tensors="pt")
        
        # Text processing
        if hasattr(self.processor, 'text_processor'):
            # Use the custom processor
            if self.text_encoder_type == 'sbert':
                # SBERT typically uses max_length=128
                text_inputs = self.processor.text_processor(
                    text=[node_text], 
                    padding='max_length', 
                    truncation=True, 
                    max_length=128, 
                    return_tensors="pt"
                )
            else:
                # CLIP uses max_length=77
                text_inputs = self.processor.text_processor(
                    text=[node_text], 
                    padding='max_length', 
                    truncation=True, 
                    max_length=77, 
                    return_tensors="pt"
                )
        else:
            # Use the standard CLIP processor
            text_inputs = self.processor(
                text=[node_text], 
                padding='max_length', 
                truncation=True, 
                max_length=77, 
                return_tensors="pt"
            )
        
        return image_inputs['pixel_values'], text_inputs


class QformerEvalDataCollator:
    def __call__(self, examples):
        # Node image inputs
        node_image_inputs = torch.cat([example["node_image_inputs"] for example in examples])
        
        # Node text inputs
        node_text_input_ids = torch.cat([example["node_text_inputs"]['input_ids'] for example in examples])
        
        # Check whether the batch contains attention_mask
        if 'attention_mask' in examples[0]["node_text_inputs"]:
            node_text_attention_mask = torch.cat([example["node_text_inputs"]['attention_mask'] for example in examples])
            node_text_inputs = {
                'input_ids': node_text_input_ids,
                'attention_mask': node_text_attention_mask
            }
        else:
            node_text_inputs = {
                'input_ids': node_text_input_ids
            }
        
        # Build the return dict
        batch = {
            "node_image_inputs": node_image_inputs,
            "node_text_inputs": node_text_inputs
        }
        
        return batch
