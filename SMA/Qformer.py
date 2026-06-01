from torch import nn
import torch
from transformers import CLIPVisionModel, CLIPTextModel, CLIPModel, AutoModel, PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel
import numpy as np
import pdb
import torch.nn.functional as F

class QformerOutput(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dense1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.dense2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.act_fn = torch.nn.GELU()
        self.LayerNorm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.act_fn(hidden_states)

        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MuliHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, is_cross_att):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.out_channels = out_dim * num_heads
        self.is_cross_att = is_cross_att

        self.Q = nn.Linear(in_dim, out_dim*num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim*num_heads, bias=True)
        self.V = nn.Linear(in_dim, out_dim*num_heads, bias=True)

        self.scale_constant = np.sqrt(self.out_dim)

    def forward(self, Q_query, inputs):
        batch_size, _, _ = Q_query.shape

        #Q_h: [batch_size, num_nodes, out_dim*num_heads]
        if self.is_cross_att:
            Q_h = self.Q(Q_query)
            K_h = self.K(inputs)
            V_h = self.V(inputs)
        else:
            Q_h = self.Q(inputs)
            K_h = self.K(inputs)
            V_h = self.V(inputs)

        # transpose to [batch_size, num_heads, num_nodes, out_dim]
        Q_h = Q_h.view(batch_size, -1, self.num_heads, self.out_dim).transpose(1,2)
        K_h = K_h.view(batch_size, -1, self.num_heads, self.out_dim).transpose(1,2)
        V_h = V_h.view(batch_size, -1, self.num_heads, self.out_dim).transpose(1,2)

        score = torch.matmul(Q_h, K_h.transpose(-2, -1)) / self.scale_constant

        attention = torch.softmax(score, dim=-1) #[batch_size, num_heads, num_nodes, num_nodes]

        
        context = torch.matmul(attention, V_h)
        attention = attention.transpose(1, 2)
        context = context.transpose(1,2)

        att_out = context.contiguous().reshape(batch_size, -1, self.out_channels)

        return att_out

class QformerLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, layer_id, cross_att_frequency, dropout=0, layer_norm=False):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.layer_id = layer_id
        self.layer_norm = layer_norm

        self.SelfAttention = MuliHeadAttentionLayer(in_dim= in_dim, out_dim= out_dim//num_heads, num_heads=num_heads, is_cross_att=False)
        if layer_id % cross_att_frequency == 0:
            self.has_cross_attention = True
            self.CrossAttention = MuliHeadAttentionLayer(in_dim=in_dim, out_dim=out_dim//num_heads, num_heads= num_heads, is_cross_att=True)
        else:
            self.has_cross_attention = False

        self.FFN = QformerOutput(in_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            self.layer_norm2 = nn.LayerNorm(out_dim)

    def forward(self, Q_query, bert_cls):

        if self.has_cross_attention == True:
            outputs = self.CrossAttention(Q_query, bert_cls)
        else:
            outputs = self.SelfAttention(Q_query, bert_cls)

        layer_out = self.FFN(outputs)

        return layer_out


class MultiModal_QformerModel(nn.Module):
    def __init__(self, num_queries, in_dim, out_dim, num_heads, freeze_encoders, num_layers, cross_att_frequency, text_encoder_type, vision_encoder_type="clip"):
        super().__init__()
        self.num_queries = num_queries
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.cross_att_frequency = cross_att_frequency
        self.text_encoder_type = text_encoder_type
        self.vision_encoder_type = vision_encoder_type

        # Vision encoder selection
        if vision_encoder_type == "clip":
            self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
            self.vision_hidden_size = 1024  # CLIP vision encoder hidden size
        elif vision_encoder_type == "dino":
            from transformers import AutoModel
            try:
                self.vision_encoder = AutoModel.from_pretrained("facebook/dinov2-base")
                self.vision_hidden_size = 768  # DINOv2-base hidden size
            except Exception as e:
                print(f"Error loading DINO model: {e}")
                # Fallback loading method
                import timm
                print("Loading DINOv2 model with timm...")
                self.vision_encoder = timm.create_model('vit_base_patch16_dinov2', pretrained=True)
                self.vision_hidden_size = 768
                
                # Add forward method to match Hugging Face interface
                original_forward = self.vision_encoder.forward
                def new_forward(pixel_values, **kwargs):
                    if hasattr(pixel_values, 'pixel_values'):
                        pixel_values = pixel_values.pixel_values
                    
                    # Make sure input shape is correct
                    if len(pixel_values.shape) == 5:  # [B, K, C, H, W]
                        B, K, C, H, W = pixel_values.shape
                        pixel_values = pixel_values.view(B*K, C, H, W)
                    
                    features = original_forward(pixel_values)
                    
                    # Wrap output to match HF model format
                    class DummyOutput:
                        def __init__(self, hidden_states):
                            self.last_hidden_state = hidden_states
                    
                    if isinstance(features, dict) and 'last_hidden_state' in features:
                        return features
                    else:
                        return DummyOutput(features)
                
                self.vision_encoder.forward = new_forward
        else:
            raise ValueError(f"Unsupported vision encoder type: {vision_encoder_type}")
            
        # Text encoder selection
        if self.text_encoder_type == 'clip':
            self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
            self.text_hidden_size = 768  # CLIP text encoder hidden size
        elif self.text_encoder_type == 'sbert':
            from transformers import AutoModel
            self.text_encoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            self.text_hidden_size = 768  # SBERT hidden size
        else:
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")
        
        # Projection layers
        if vision_encoder_type == "dino":
            self.vision_projector = nn.Linear(self.vision_hidden_size, 1024)
            
        self.text_projector = nn.Linear(self.text_hidden_size, 1024)

        if freeze_encoders:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.query_tokens = nn.Parameter(torch.randn(1, self.num_queries, in_dim))
        self.layers = nn.ModuleList(
            [QformerLayer(in_dim, in_dim, out_dim, num_heads, (layer_idx+1), cross_att_frequency) for layer_idx in range(num_layers)]
        )
        

    def forward(self, image_pixel, text_input_ids, text_attention_mask):
        # Image processing
        if self.vision_encoder_type == "clip":
            image_token = self.vision_encoder(image_pixel).last_hidden_state
        elif self.vision_encoder_type == "dino":
            image_token = self.vision_encoder(image_pixel).last_hidden_state
            # Project DINO output to the standard dimension
            image_token = self.vision_projector(image_token)

        # Text processing - depends on the text encoder type
        if self.text_encoder_type == 'clip':
            text_token = self.text_encoder(text_input_ids, text_attention_mask).last_hidden_state
        elif self.text_encoder_type == 'sbert':
            # SBERT uses the same input format as BERT
            text_outputs = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
                output_hidden_states=True
            )
            # Take all tokens from the last hidden layer
            text_token = text_outputs.last_hidden_state
        
        # Project text embeddings to the standard dimension
        text_token = self.text_projector(text_token)

        batch_size = image_token.size(0)
        query = self.query_tokens.expand(batch_size, -1, -1)

        for layer in self.layers:
            if layer.has_cross_attention:
                text_token = layer.SelfAttention(query, text_token)
                image_token = layer.SelfAttention(query, image_token)
                x = torch.cat((image_token, text_token), dim=1)
                x = layer(query, x)
                text_token = layer.FFN(text_token)
                image_token = layer.FFN(image_token)
                query = x
            else:
                text_token = layer(query, text_token)
                image_token = layer(query, image_token)

        return x

        
class MultiModal_QformerConfig(PretrainedConfig):
    model_type = "multimodal_qformer"
    
    def __init__(
        self,
        num_queries=32,
        in_dim=1024,
        out_dim=1024,
        num_heads=8,
        num_layers=6,
        freeze_encoders=True,
        cross_att_frequency=3,
        text_encoder_type="clip",
        vision_encoder_type="clip",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.freeze_encoders = freeze_encoders
        self.cross_att_frequency = cross_att_frequency
        self.text_encoder_type = text_encoder_type
        self.vision_encoder_type = vision_encoder_type

class MultiModal_Qformer(PreTrainedModel):
    config_class = MultiModal_QformerConfig
    def __init__(self, config):
        PreTrainedModel.__init__(self, config)
        self.config = config
        self.model = MultiModal_QformerModel(
            config.num_queries, 
            config.in_dim, 
            config.out_dim, 
            config.num_heads, 
            config.freeze_encoders, 
            config.num_layers, 
            config.cross_att_frequency, 
            config.text_encoder_type,
            config.vision_encoder_type
        )

    def compute_loss(self, batch_central_nodes, batch_neighbor_nodes, temperature=0.5, global_ids=None):
        
        batch_central_nodes = torch.mean(batch_central_nodes, dim=1)
        batch_size, hidden_dim = batch_central_nodes.shape
        _, k, _ = batch_neighbor_nodes.shape

        positive_samples = batch_neighbor_nodes.view(-1, hidden_dim)
        positive_similarity = F.cosine_similarity(batch_central_nodes.unsqueeze(1).expand(-1, k, -1).reshape(-1, hidden_dim), positive_samples).view(batch_size, k)
        positive_similarity = torch.exp(positive_similarity)

        negative_samples = batch_central_nodes.unsqueeze(1).repeat(1, batch_size, 1)
        negative_similarity = F.cosine_similarity(negative_samples, batch_central_nodes.unsqueeze(0).repeat(batch_size, 1, 1), dim=2)
        negative_similarity = negative_similarity / temperature
        negative_similarity = torch.exp(negative_similarity)
        if global_ids is not None:
            import numpy as np
            ids_np = np.array(global_ids)
            mask = (ids_np[:, None] != ids_np[None, :]).astype(np.float32)  # [B, B]
            mask = torch.tensor(mask, device=negative_similarity.device)
            negative_similarity = negative_similarity * mask  # apply mask

    # -- Denominator sum
        negative_similarity = negative_similarity.sum(dim=1) + 1e-8  # [B]

    # -- Final contrastive loss
        loss = -torch.log(positive_similarity.sum(dim=1) / negative_similarity).mean()
        return loss

    def forward(self, node_image_inputs, node_text_inputs, neighbor_image_inputs=None, neighbor_text_inputs=None, global_ids= None, **kwargs):
        batch_global_ids = kwargs.get("global_ids", None)
        node_query = self.model(node_image_inputs, node_text_inputs['input_ids'], node_text_inputs['attention_mask'])
        if neighbor_image_inputs is None or neighbor_text_inputs is None:
            return node_query
        batch_size, num_neighbors, text_len = neighbor_text_inputs['input_ids'].shape
        # Flatten the neighbor inputs
        flatten_neighbor_text_input_ids = neighbor_text_inputs['input_ids'].view(-1, text_len)
        flatten_neighbor_text_inputs_attention_mask = neighbor_text_inputs['attention_mask'].view(-1, text_len)
        B, K, C, H, W = neighbor_image_inputs.shape
        flatten_neighbor_image_inputs = neighbor_image_inputs.view(B * K, C, H, W)

        
        neighbor_queries = self.model(flatten_neighbor_image_inputs, flatten_neighbor_text_input_ids, flatten_neighbor_text_inputs_attention_mask)
        neighbor_queries = neighbor_queries.view(batch_size, num_neighbors, -1, self.config.in_dim)
        pooled_neighbor_queries = torch.mean(neighbor_queries, dim=2)
        loss = self.compute_loss(node_query, pooled_neighbor_queries, temperature=0.5, global_ids=batch_global_ids)
        return {
            'loss': loss,
            'node_query': node_query
        }

AutoConfig.register("multimodal_qformer", MultiModal_QformerConfig)
AutoModel.register(MultiModal_QformerConfig, MultiModal_Qformer)