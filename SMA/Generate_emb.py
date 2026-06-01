from transformers import AutoModel, CLIPProcessor, AutoImageProcessor, AutoTokenizer
from Qformer import MultiModal_Qformer
from utils import QformerEvalDataset, QformerEvalDataCollator
import sys
sys.path.append("..")
from data_utils.data_loader import load_data
from torch.utils.data import DataLoader
import torch
import pdb
from tqdm import tqdm
import argparse
import os

# Unified processor wrapper class
class UnifiedProcessor:
    """Wrapper that unifies the CLIP, DINO and SBERT processor interfaces."""
    
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    parser.add_argument('--dataset', type=str, default="Movies")
    parser.add_argument('--full_text', type=bool, default=True)
    parser.add_argument('--vision_encoder_type', type=str, default="clip", choices=["clip", "dino"], help="Vision encoder type")
    parser.add_argument('--text_encoder_type', type=str, default="clip", choices=["clip", "sbert"], help="Text encoder type")
    parser.add_argument('--model_path', type=str, default="./output/Reddit", help="Path to the pretrained model")
    parser.add_argument('--output_name', type=str, default="query_token", help="Output embedding file name")
    parser.add_argument('--batch_size', type=int, default=16, help="Inference batch size")

    args = parser.parse_args()

    dataset = args.dataset
    full_text = args.full_text
    vision_encoder_type = args.vision_encoder_type
    text_encoder_type = args.text_encoder_type
    
    # Pick the processor according to the requested encoder type
    if vision_encoder_type == "clip" and text_encoder_type == "clip":
        # Standard CLIP processor
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        print("Using CLIP vision encoder + CLIP text encoder")
    elif vision_encoder_type == "clip" and text_encoder_type == "sbert":
        # CLIP vision processor + SBERT text processor
        vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14").image_processor
        text_processor = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        processor = UnifiedProcessor(vision_processor, text_processor)
        print("Using CLIP vision encoder + SBERT text encoder")
    elif vision_encoder_type == "dino" and text_encoder_type == "clip":
        # DINO vision processor + CLIP text processor
        try:
            vision_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            text_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14").tokenizer
            processor = UnifiedProcessor(vision_processor, text_processor)
            print("Using DINO vision encoder + CLIP text encoder")
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
            text_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14").tokenizer
            processor = UnifiedProcessor(vision_processor, text_processor)
            print("Using simple image processor + CLIP text encoder")
    elif vision_encoder_type == "dino" and text_encoder_type == "sbert":
        # DINO vision processor + SBERT text processor
        try:
            vision_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            text_processor = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            processor = UnifiedProcessor(vision_processor, text_processor)
            print("Using DINO vision encoder + SBERT text encoder")
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
            print("Using simple image processor + SBERT text encoder")
    else:
        raise ValueError(f"Unsupported encoder combination: vision={vision_encoder_type}, text={text_encoder_type}")

    # Load the pretrained model
    print(f"Loading model from path: {args.model_path}")
    model = MultiModal_Qformer.from_pretrained(args.model_path).to(device)
    
    # Load the data
    file_path = f"../dataset/{dataset}/"
    print(f"Loading dataset: {dataset}")
    data, text = load_data(dataset_name=dataset)
    
    # Build the evaluation dataset
    eval_dataset = QformerEvalDataset(
        data.images, 
        text, 
        processor, 
        vision_encoder_type=vision_encoder_type,
        text_encoder_type=text_encoder_type
    )
    
    # Build the data loader
    eval_data_collator = QformerEvalDataCollator()
    dataloader = DataLoader(
        dataset=eval_dataset, 
        batch_size=args.batch_size, 
        collate_fn=eval_data_collator, 
        shuffle=False, 
        num_workers=8
    )
    
    # Generate embeddings
    token_list = []
    model.eval()
    
    print(f"Start generating embeddings for the {dataset} dataset...")
    for batch in tqdm(dataloader, desc="Processing batches"):
        node_image_inputs = batch['node_image_inputs'].to(device)
        node_text_inputs = {k: v.to(device) for k, v in batch['node_text_inputs'].items()}
        
        with torch.no_grad():
            node_outputs = model(node_image_inputs, node_text_inputs).cpu()
            token_list.append(node_outputs)
    
    # Concatenate all embeddings
    query_token = torch.cat(token_list, dim=0)
    
    # Save embeddings
    output_filename = f"{args.output_name}_{vision_encoder_type}_{text_encoder_type}.pt"
    output_path = f"../dataset/{dataset}/{output_filename}"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the embedding tensor
    torch.save(query_token, output_path)
    print(f"Embeddings saved to: {output_path}")
    print(f"Embedding shape: {query_token.shape}")
