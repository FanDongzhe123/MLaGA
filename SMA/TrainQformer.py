import sys
sys.path.append("..")
import torch
from data_utils.data_loader import load_data
from utils import QformerDataset, QformerDataCollator, neighbor_sampler
import pdb
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import transformers
import torch
from torch.utils.data import Dataset
from transformers import Trainer, PretrainedConfig, CLIPProcessor, CLIPTokenizer
from torch.utils.data import DataLoader
from Qformer import MultiModal_Qformer
from PIL import Image


# Unified processor wrapper class
class UnifiedProcessor:
    """Wrapper that unifies the CLIP and DINO processor interfaces."""
    
    def __init__(self, vision_processor, text_processor):
        self.vision_processor = vision_processor
        self.text_processor = text_processor
    
    def __call__(self, images=None, text=None, **kwargs):
        if images is not None and text is None:
            # Image processing
            result = self.vision_processor(images=images, **kwargs)
            return result
        elif text is not None and images is None:
            # Text processing
            result = self.text_processor(text=text, **kwargs)
            return result
        else:
            raise ValueError("Please call image or text processing separately, not both at the same time")


@dataclass
class ModelArguments(PretrainedConfig):
    num_queries: Optional[int] = field(default=32)
    in_dim: Optional[int] = field(default=1024)
    out_dim: Optional[int] = field(default=1024)
    num_heads: Optional[int] = field(default=8)
    num_layers: Optional[int] = field(default=6)
    freeze_encoders: Optional[bool] = field(default=True)
    cross_att_frequency: Optional[int] = field(default=3)
    text_encoder_type: Optional[str] = field(default="clip")
    vision_encoder_type: Optional[str] = field(default="clip")  # newly added parameter


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default="Movies")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="./output/Movies")
    per_device_train_batch_size: Optional[int] = field(default=16)
    num_train_epochs: Optional[int] = field(default=1)
    lr_scheduler_type: Optional[str] = field(default="cosine")
    remove_unused_columns: Optional[bool] = field(default=False)
    logging_strategy: Optional[str] = field(default="steps")
    logging_steps: Optional[int] = field(default=1)
    logging_first_step: Optional[bool] = field(default=True)
    learning_rate: Optional[float] = field(default=1e-5)
    dataloader_num_workers: Optional[int] = field(default=8)
    warmup_steps: Optional[int] = field(default=100)
    save_strategy: Optional[str] = field(default="epoch")


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                neighbor_list,
                                image_paths,
                                texts,
                                global_ids,
                                vision_encoder_type="clip",
                                text_encoder_type="clip") -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = QformerDataset(
        neighbor_list, 
        image_paths, 
        texts, 
        tokenizer, 
        vision_encoder_type=vision_encoder_type,
        text_encoder_type=text_encoder_type,
        global_ids=global_ids
    )
    data_collator = QformerDataCollator()

    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    parser = transformers.HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    if data_args.dataset_name == 'mix':
        dataset_names = ['Movies','VideoGames','Arts', 'RedditS','Health', 'CD', 'Beauty', 'Art500K']

        all_data       = []
        texts          = []   
        image_paths    = []
        neighbor_list  = []
        global_ids     = []
        repeat_times = 1

        node_count = 0
        for name in dataset_names:
            temp_data, temp_text = load_data(dataset_name=name)
            num_nodes            = len(temp_text)
            temp_neighbors       = neighbor_sampler(temp_data, samples_size=5)
            temp_images          = temp_data.images

            n_reps = 1 if name=='Goodreads' else repeat_times
            for n_rep in range(n_reps):
                print(f"Processing {name} dataset for {n_rep} times, dataset size: {num_nodes}")
                texts.extend(temp_text)

                # data list (kept here in case it is still used)
                all_data.append(temp_data)

                # image paths
                image_paths.extend(temp_images)

                # shift neighbor indices to the global index space
                shifted = [[nbr+node_count for nbr in neighs]
                        for neighs in temp_neighbors]
                neighbor_list.extend(shifted)

                # build global ids
                for orig_idx in range(num_nodes):
                    global_ids.append(f"{name}_{orig_idx}")

                node_count += num_nodes
        print(f"Total number of nodes: {node_count}")

    else:
        temp_data, temp_text = load_data(dataset_name=data_args.dataset_name)
        all_data      = [temp_data]
        texts         = temp_text
        image_paths   = temp_data.images
        neighbor_list = neighbor_sampler(temp_data, samples_size=5)
        global_ids    = [f"{data_args.dataset_name}_{i}" for i in range(len(temp_text))]

    # Load the processor according to the selected encoder type
    if model_args.vision_encoder_type == "clip" and model_args.text_encoder_type == "clip":
        # Standard CLIP processor
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        print(f"Using CLIP vision encoder + CLIP text encoder")
    elif model_args.vision_encoder_type == "clip" and model_args.text_encoder_type == "sbert":
        # CLIP vision processor + SBERT text processor
        from transformers import CLIPImageProcessor, AutoTokenizer
        vision_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        text_processor = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        processor = UnifiedProcessor(vision_processor, text_processor)
        print(f"Using CLIP vision encoder + SBERT text encoder")
    elif model_args.vision_encoder_type == "dino" and model_args.text_encoder_type == "clip":
        # DINO vision processor + CLIP text processor
        try:
            from transformers import AutoImageProcessor, CLIPTokenizer
            vision_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            text_processor = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            processor = UnifiedProcessor(vision_processor, text_processor)
            print(f"Using DINO vision encoder + CLIP text encoder")
        except Exception as e:
            print(f"Error loading DINO processor: {e}")
            # Fallback: use a simple image processor
            from torchvision import transforms
            
            class SimpleImageProcessor:
                def __init__(self):
                    self.transforms = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                
                def __call__(self, images=None, **kwargs):
                    if images is not None:
                        if not isinstance(images, list):
                            images = [images]
                        
                        # Process all images
                        pixel_values = []
                        for img in images:
                            pixel_values.append(self.transforms(img))
                        
                        # Stack all image tensors
                        pixel_values = torch.stack(pixel_values)
                        return {'pixel_values': pixel_values}
                    return {}
            
            vision_processor = SimpleImageProcessor()
            text_processor = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            processor = UnifiedProcessor(vision_processor, text_processor)
            print(f"Using simple image processor + CLIP text encoder")
    elif model_args.vision_encoder_type == "dino" and model_args.text_encoder_type == "sbert":
        # DINO vision processor + SBERT text processor
        try:
            from transformers import AutoImageProcessor, AutoTokenizer
            vision_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            text_processor = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            processor = UnifiedProcessor(vision_processor, text_processor)
            print(f"Using DINO vision encoder + SBERT text encoder")
        except Exception as e:
            print(f"Error loading DINO processor: {e}")
            # Fallback: use a simple image processor
            from torchvision import transforms
            
            class SimpleImageProcessor:
                def __init__(self):
                    self.transforms = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                
                def __call__(self, images=None, **kwargs):
                    if images is not None:
                        if not isinstance(images, list):
                            images = [images]
                        
                        # Process all images
                        pixel_values = []
                        for img in images:
                            pixel_values.append(self.transforms(img))
                        
                        # Stack all image tensors
                        pixel_values = torch.stack(pixel_values)
                        return {'pixel_values': pixel_values}
                    return {}
            
            vision_processor = SimpleImageProcessor()
            text_processor = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            processor = UnifiedProcessor(vision_processor, text_processor)
            print(f"Using simple image processor + SBERT text encoder")
    else:
        raise ValueError(f"Unsupported encoder combination: vision={model_args.vision_encoder_type}, text={model_args.text_encoder_type}")

    # Build the model
    model = MultiModal_Qformer(model_args)

    # Build the data module
    data_module = make_supervised_data_module(
        tokenizer=processor,
        neighbor_list=neighbor_list,
        image_paths=image_paths,
        texts=texts,
        global_ids=global_ids,
        vision_encoder_type=model_args.vision_encoder_type,
        text_encoder_type=model_args.text_encoder_type
    )
    
    # Build the trainer
    QformerTrainer = Trainer(
        model=model,
        args=train_args,
        tokenizer=None,
        **data_module
    )
    
    # Train the model
    QformerTrainer.train()
    QformerTrainer.save_state()
    QformerTrainer.save_model(train_args.output_dir)