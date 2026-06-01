
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTextModel
import sys
sys.path.append("..")
from data_utils.data_loader import load_data
from PIL import Image
import torch
import torch.nn.functional as F
from utils import CLIPDataset, init_path
from torch.utils.data import DataLoader
from tqdm import tqdm
import pdb

datasets = ["Wikiweb2m"]
# dataset_name = "Movies"
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
# vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
# text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
for dataset_name in datasets:
    data, text = load_data(dataset_name=dataset_name)
    dataset = CLIPDataset(data.images, text, processor, dataset_name=dataset_name)
    data_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=False)
    
    image_path = f'../dataset/{dataset_name}/clip_image_new.pt'
    text_path = f'../dataset/{dataset_name}/clip_text_new.pt'
    # image_all_path = f'./Embeddings/{dataset_name}/Image_feature/{dataset_name}-image-all.pt'
    # image_clip_path = f'./Embeddings/{dataset_name}/Image_feature/{dataset_name}-image-clip.pt'
    # id_path = f'./Embeddings/{dataset_name}/Image_feature/{dataset_name}_id.pt'
    # text_cls_path = f'./Embeddings/{dataset_name}/Image_feature/{dataset_name}-text-cls.pt'
    # text_path = f'./Embeddings/{dataset_name}/Text_feature/{dataset_name}-text-normal.pt'
    with torch.no_grad():
        # vision_encoder.eval()
        model.eval()
        # image_cls_list = []
        # image_all_list = []
        # text_cls_list = []
        image_list = []
        text_list = []
        count = 0
        for batch in tqdm(data_loader, desc=f"Generating {dataset_name} Features"):
            image, text = batch
            # text = batch
            image = image.squeeze(1).to(device)
            input_ids = text['input_ids'].squeeze(1).to(device)
            attention_mask = text['attention_mask'].squeeze(1).to(device)

            text_emb = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask).cpu()
            image_emb = model.get_image_features(image).cpu()

            image_list.append(image_emb)
            text_list.append(text_emb)

            # outputs = vision_encoder(image)
            # last_hidden_state = outputs.last_hidden_state
            # image_cls = last_hidden_state[:, 0, :].cpu()
            # last_hidden_state = last_hidden_state.cpu()
            # text_embedding = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            # outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            # last_hidden_state = outputs.last_hidden_state
            # text_cls = last_hidden_state[:, 0, :].cpu()

            # image_cls_list.append(image_cls)
            # image_all_list.append(last_hidden_state)
            # text_cls_list.append(text_cls)
        # torch.save(dataset.fail_idx, init_path(id_path))
        image_feature = torch.cat(image_list, dim=0)
        torch.save(image_feature, init_path(image_path))
        text_feature = torch.cat(text_list, dim=0)
        torch.save(text_feature, init_path(text_path))
                
        # image_cls_feature = torch.cat(image_cls_list, dim=0)
        # torch.save(image_cls_feature, init_path(image_cls_path))
        # image_all_feature = torch.cat(image_all_list, dim=0)
        # torch.save(image_all_feature, init_path(image_all_path))
        # text_feature = torch.cat(text_embedding_list, dim=0)
        # torch.save(text_cls_feature, init_path(text_cls_path))