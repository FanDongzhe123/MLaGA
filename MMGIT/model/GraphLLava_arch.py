from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import re
import pdb

# from .multimodal_encoder.builder import build_vision_tower
# from .multimodal_projector.builder import build_vision_projector

from ..constants import IGNORE_INDEX, GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN, DEFAULT_GRAPH_PAD_ID, IMAGE_TOKEN_INDEX, TEXT_TOKEN_INDEX, NODE_TOKEN_INDEX

class CrossTaskAttention(nn.Module):
    """
    Cross-task attention module that enables knowledge transfer between task-specific projectors.
    Supports multiple layers of transformer blocks with customizable depth.
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, ffn_dim=None, num_layers=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_layers = num_layers
        
        # Build the multi-layer attention modules
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            # QKV projections
            q_proj = nn.Linear(hidden_dim, hidden_dim)
            k_proj = nn.Linear(hidden_dim, hidden_dim)
            v_proj = nn.Linear(hidden_dim, hidden_dim)
            out_proj = nn.Linear(hidden_dim, hidden_dim)
            
            # Layer normalization
            attn_layer_norm = nn.LayerNorm(hidden_dim)
            ffn_layer_norm = nn.LayerNorm(hidden_dim)
            
            # FFN
            if ffn_dim is None:
                ffn_dim = hidden_dim * 4  # FFN dim used in standard Transformer
                
            ffn = nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, hidden_dim),
                nn.Dropout(dropout)
            )
            
            # Pack all components into the current layer
            layer = nn.ModuleDict({
                'q_proj': q_proj,
                'k_proj': k_proj,
                'v_proj': v_proj,
                'out_proj': out_proj,
                'attn_layer_norm': attn_layer_norm,
                'ffn_layer_norm': ffn_layer_norm,
                'ffn': ffn,
                'dropout': nn.Dropout(dropout)
            })
            
            self.layers.append(layer)


    # original code
    def forward(self, query, key_value, task_type=None):
        """
        Args:
            query: [seq_len, hidden_dim] or [batch_size, seq_len, hidden_dim] - Features from primary projector
            key_value: [seq_len, hidden_dim] or [batch_size, seq_len, hidden_dim] - Features from secondary projector
            task_type: 'nc' or 'lp' - Type of task for handling padding tokens
        Returns:
            output: [seq_len, hidden_dim] or [batch_size, seq_len, hidden_dim] - Enhanced features with residual connections
        """
        # Make sure both inputs share the same dtype
        if query.dtype != key_value.dtype:
            key_value = key_value.to(dtype=query.dtype)
        
        # Check input dims so we accept either 2D or 3D inputs
        if query.dim() == 2:
            # Handle inputs shaped [seq_len, hidden_dim]
            # Add a batch dimension
            query = query.unsqueeze(0)  # [1, seq_len, hidden_dim]
            key_value = key_value.unsqueeze(0)  # [1, seq_len, hidden_dim]
            is_2d_input = True
        else:
            is_2d_input = False
        
        # Now it is safe to unpack the shapes
        batch_size, seq_len, hidden_dim = query.shape
        
        # Build a padding mask for NC tasks where the second half tokens are masked out
        padding_mask = None
        if task_type == 'nc':
            # For NC tasks, of the 64 tokens the last 32 are padding
            padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=query.device)
            padding_mask[:, seq_len//2:] = True  # second half is padding
        all_layer_attn_weights = []
        # Multi-layer processing
        x = query
        for i in range(self.num_layers):
            layer = self.layers[i]
            
            # Save the residual for this layer
            residual = x
            
            # Apply layer normalization
            query_norm = layer['attn_layer_norm'](x)
            key_value_norm = layer['attn_layer_norm'](key_value if i == 0 else x)
            
            # Project Q, K, V
            q = layer['q_proj'](query_norm).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = layer['k_proj'](key_value_norm).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = layer['v_proj'](key_value_norm).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Compute attention weights
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # Apply the padding mask
            if padding_mask is not None:
                # Expand the padding mask to fit the attention matrix shape [batch_size, 1, seq_len, seq_len]
                attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)
                
                # Make sure the mask matches the attention dtype
                # This sets attention weights for padded positions to -inf
                attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
            
            # Apply softmax to obtain the attention weights
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = layer['dropout'](attn_weights)
            
            # Apply attention weights to the values
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
            attn_output = layer['out_proj'](attn_output)
            attn_output = layer['dropout'](attn_output)
            
            # First residual connection
            attn_output = attn_output + residual
            
            # Second residual branch - FFN
            residual = attn_output
            
            # Apply layer normalization
            ffn_output = layer['ffn_layer_norm'](attn_output)
            ffn_output = layer['ffn'](ffn_output)
            
            # Second residual connection
            x = ffn_output + residual
            
            # For NC tasks, apply the padding mask again
            if padding_mask is not None:
                # Make sure the mask matches the output dtype
                # Use the mask to zero out padding positions
                x = x * (~padding_mask).unsqueeze(-1).to(dtype=x.dtype)
        
        # If the input was 2D, drop the batch dimension before returning
        if is_2d_input:
            x = x.squeeze(0)  # [seq_len, hidden_dim]

        return x

def build_graph_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    hidden_dim = getattr(config, 'word_embed_proj_dim', getattr(config, 'hidden_size', 'linear'))

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, hidden_dim)
    mlp_gelu_match = re.match(r'^(\d+)-layer-mlp$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, hidden_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_dim, hidden_dim))
        return nn.Sequential(*modules)
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')

def build_shared_first_layer(config, delay_load=False, **kwargs):
    """Build the shared first-layer MLP."""
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    hidden_dim = getattr(config, 'word_embed_proj_dim', getattr(config, 'hidden_size', 'linear'))

    if projector_type == 'linear':
        # For a linear projector, return a plain linear layer
        return nn.Linear(config.mm_hidden_size, hidden_dim)
    
    mlp_gelu_match = re.match(r'^(\d+)-layer-mlp$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        if mlp_depth <= 1:
            # Single-layer MLP, return as is
            return nn.Linear(config.mm_hidden_size, hidden_dim)
        
        # Only build the first layer
        return nn.Sequential(
            nn.Linear(config.mm_hidden_size, hidden_dim),
            nn.GELU()
        )
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')


def build_task_specific_layer(config, delay_load=False, **kwargs):
    """Build the task-specific second layer."""
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    hidden_dim = getattr(config, 'word_embed_proj_dim', getattr(config, 'hidden_size', 'linear'))

    mlp_gelu_match = re.match(r'^(\d+)-layer-mlp$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        if mlp_depth <= 1:
            # Single-layer MLP, return an identity mapping
            return nn.Identity()
        
        # Only build the second layer
        return nn.Linear(hidden_dim, hidden_dim)
    else:
        # For a linear projector, return an identity mapping
        return nn.Identity()

def build_task_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    hidden_dim = getattr(config, 'word_embed_proj_dim', getattr(config, 'hidden_size', 'linear'))

    if projector_type == 'linear':
        return nn.Linear(hidden_dim, hidden_dim)
    
    mlp_gelu_match = re.match(r'^(\d+)-layer-mlp$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(hidden_dim, hidden_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_dim, hidden_dim))
        return nn.Sequential(*modules)
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')
    
def build_image_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    hidden_dim = getattr(config, 'word_embed_proj_dim', getattr(config, 'hidden_size', 'linear'))

    if projector_type == 'linear':
        return nn.Linear(config.mm_image_hidden_size, hidden_dim)
    mlp_gelu_match = re.match(r'^(\d+)-layer-mlp$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_image_hidden_size, hidden_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_dim, hidden_dim))
        return nn.Sequential(*modules)
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')

def build_text_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    hidden_dim = getattr(config, 'word_embed_proj_dim', getattr(config, 'hidden_size', 'linear'))

    if projector_type == 'linear':
        return nn.Linear(config.mm_text_hidden_size, hidden_dim)
    mlp_gelu_match = re.match(r'^(\d+)-layer-mlp$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_text_hidden_size, hidden_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_dim, hidden_dim))
        return nn.Sequential(*modules)
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')
    
def build_node_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    hidden_dim = getattr(config, 'word_embed_proj_dim', getattr(config, 'hidden_size', 'linear'))

    if projector_type == 'linear':
        return nn.Linear(config.mm_node_hidden_size, hidden_dim)
    mlp_gelu_match = re.match(r'^(\d+)-layer-mlp$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_node_hidden_size, hidden_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_dim, hidden_dim))
        return nn.Sequential(*modules)
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')



class GraphLLaVAMetaModel:

    def __init__(self, config):
        super(GraphLLaVAMetaModel, self).__init__(config)

        if hasattr(config, "mm_hidden_size"):
            if getattr(config, 'is_cross_task_attention', False):

                hidden_dim = getattr(config, 'hidden_size', 1024)
                self.mm_projector_nc = build_graph_projector(config)
                self.mm_projector_lp = build_graph_projector(config)
                

                num_heads = getattr(config, 'cross_attn_heads', 8)
                attn_dropout = getattr(config, 'attn_dropout', 0.1)
                ffn_dim = getattr(config, 'ffn_dim', hidden_dim * 4)
                
                self.cross_task_attention = CrossTaskAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    ffn_dim=ffn_dim
                )
                
                print(f"Initialized cross-task attention projector with {num_heads} heads")
            else:
                self.mm_projector_graph = build_graph_projector(config)

        if hasattr(config, "mm_text_hidden_size"):
            self.mm_projector_text = build_text_projector(config)

        if hasattr(config, "mm_image_hidden_size"):
            self.mm_projector_image = build_image_projector(config)

        if hasattr(config, "mm_node_hidden_size"):
            self.mm_projector_node = build_node_projector(config)
        if hasattr(config, "mm_use_graph_special_token") and getattr(config, 'mm_use_graph_special_token', False):
            self.special_token_emb = self.build_special_tokens()

    def initialize_graph_modules(self, model_args, fsdp=None):
        """
        Initialize graph modules and projectors, with support for loading pretrained
        projectors when using a pretrained LoRA model.
        """
        # Get configuration options
        pretrain_mm_mlp_adapter = getattr(model_args, 'pretrain_mm_mlp_adapter', None)
        
        # Check if pretrained LoRA path is provided
        use_pretrained_lora = getattr(model_args, 'pretrained_lora_path', None) is not None
        
        # Get paths for pretrained projectors
        pretrained_image_projector_path = getattr(model_args, 'pretrained_image_projector_path', None)
        pretrained_common_projector_path = getattr(model_args, 'pretrained_common_projector_path', None)
        
        # Update configuration with model arguments
        if hasattr(model_args, "is_data_task_general"):
            self.config.is_data_task_general = getattr(model_args, 'is_data_task_general', False)
        if hasattr(model_args, "is_cross_task_attention"):
            self.config.is_cross_task_attention = getattr(model_args, 'is_cross_task_attention', False)

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        
        # Handle graph projector initialization based on mode
        if hasattr(model_args, 'mm_hidden_size'):
            self.config.mm_hidden_size = getattr(model_args, 'mm_hidden_size')
            

            if getattr(model_args, 'is_cross_task_attention', False):

                hidden_dim = getattr(self.config, 'hidden_size', 4096)
                
                # Build the task-specific projectors
                self.mm_projector_nc = build_graph_projector(self.config)
                self.mm_projector_lp = build_graph_projector(self.config)
                
                # Build the cross-task attention module
                num_heads = getattr(model_args, 'cross_attn_heads', 8)
                attn_dropout = getattr(model_args, 'attn_dropout', 0.1)
                ffn_dim = getattr(model_args, 'ffn_dim', hidden_dim * 4)
                num_layers = getattr(model_args, 'cross_attn_layers', 1)  # number of layers
                
                self.cross_task_attention = CrossTaskAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads, 
                    dropout=attn_dropout,
                    ffn_dim=ffn_dim,
                    num_layers=num_layers  # pass through the layer count
                )
                
                print(f"Initialized cross-task attention projector with {num_heads} heads and {num_layers} layers")


            else:
                self.mm_projector_graph = build_graph_projector(self.config)
                print("Graph projector initialized in standard mode.")
        
        # Handle text projector initialization
        if hasattr(model_args, 'mm_text_hidden_size'):
            self.config.mm_text_hidden_size = getattr(model_args, 'mm_text_hidden_size')
            self.mm_projector_text = build_text_projector(self.config)
            print("Text projector initialized.")

        # Handle image projector initialization
        if hasattr(model_args, 'mm_image_hidden_size'):
            self.config.mm_image_hidden_size = getattr(model_args, 'mm_image_hidden_size')
            
            # When using pretrained LoRA and pretrained image projector is specified
            if use_pretrained_lora and pretrained_image_projector_path:
                print(f"Loading pretrained image projector from {pretrained_image_projector_path}")
                
                # Create image projector first
                self.mm_projector_image = build_image_projector(self.config)
                
                # Load pretrained weights
                try:
                    state_dict = torch.load(pretrained_image_projector_path, map_location='cpu')
                    
                    # Handle different state dict formats, including the deeply nested structure
                    if isinstance(state_dict, dict):
                        # Check for direct mm_projector_image key
                        if "mm_projector_image" in state_dict:
                            self.mm_projector_image.load_state_dict(state_dict["mm_projector_image"])
                        
                        # Check for nested structure with model.mm_projector_image
                        elif "model.mm_projector_image" in state_dict:
                            image_projector_weights = {k.replace("model.mm_projector_image.", ""): v 
                                                    for k, v in state_dict.items() 
                                                    if k.startswith("model.mm_projector_image.")}
                            self.mm_projector_image.load_state_dict(image_projector_weights)
                        
                        # Check for deeply nested structure with base_model.model.model.mm_projector_image
                        elif any(k.startswith("base_model.model.model.mm_projector_image") for k in state_dict.keys()):
                            image_projector_weights = {k.replace("base_model.model.model.mm_projector_image.", ""): v 
                                                    for k, v in state_dict.items() 
                                                    if k.startswith("base_model.model.model.mm_projector_image.")}
                            self.mm_projector_image.load_state_dict(image_projector_weights)
                        
                        # If all fails, try to load directly
                        else:
                            self.mm_projector_image.load_state_dict(state_dict)
                    else:
                        # Direct loading if it's not a dictionary
                        self.mm_projector_image.load_state_dict(state_dict)
                    
                    print(f"Successfully loaded pretrained image projector from {pretrained_image_projector_path}")
                    
                    # Image projector is trainable by default when loaded with LoRA
                    print("Image projector will be fine-tuned.")
                        
                except Exception as e:
                    print(f"Error loading pretrained image projector: {e}")
                    print("Initializing new image projector instead.")
                    self.mm_projector_image = build_image_projector(self.config)
            else:
                # Create new image projector in other cases
                self.mm_projector_image = build_image_projector(self.config)
                print("Initialized new image projector.")
        
        # Handle node projector initialization
        if hasattr(model_args, 'mm_node_hidden_size'):
            self.config.mm_node_hidden_size = getattr(model_args, 'mm_node_hidden_size')
            self.mm_projector_node = build_node_projector(self.config)
            print("Node projector initialized.")
        
        # Initialize special token embeddings if needed
        if hasattr(self.config, "mm_use_graph_special_token") and getattr(self.config, 'mm_use_graph_special_token', False):
            self.special_token_emb = self.build_special_tokens()
            print("Special token embeddings initialized.")

        # Load pretrained graph projector weights if specified (non-LoRA path)
        if pretrain_mm_mlp_adapter is not None and not use_pretrained_lora:
            try:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

                if hasattr(self, 'mm_projector_graph'):
                    self.mm_projector_graph.load_state_dict(get_w(mm_projector_weights, 'mm_projector_graph'))
                    print(f"Loaded pretrained graph projector weights from {pretrain_mm_mlp_adapter}")
                    
            except Exception as e:
                print(f"Warning: Failed to load pretrained weights: {e}")
        
        # Print initialization completion message
        print("Graph modules initialization completed.")



    def build_special_tokens(self):
        if hasattr(self.config, "mm_use_graph_special_token") and getattr(self.config, 'mm_use_graph_special_token', False):
            num_token=self.config.use_hop+2
            input_embeddings = self.get_input_embeddings().weight.data #
            input_embeddings_avg = input_embeddings.mean(dim=0, keepdim=True).unsqueeze(1).detach()
            special_token_emb=torch.nn.parameter.Parameter(data=input_embeddings_avg.repeat(num_token, 1, 1), requires_grad=True)
            return special_token_emb
        return None

class GraphLLaVAMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass
    def encode_graphs_general(self, graph_emb, graph, task_type):
        """
        Process graph embeddings with different projector architectures.
        Supports multiple modes: regular MoE, MoE with task tokens, Soft MoE,
        shared first layer, dual projector, dual projector with common, and cross-task attention.
        
        Args:
            graph_emb: Graph embeddings [batch_size, seq_len, hidden_dim]
            graph: Graph tokens with padding information
            task_type: List of task types ('nc' or 'lp')
            
        Returns:
            graph_features: Projected graph features
        """
        graph_features_list = []
        current_graph_id = 0
        model = self.get_model()
        node_attn_scores = []
        cross_task_info_flow = []
        
        if getattr(self.config, 'is_cross_task_attention', False):
            for i in range(len(task_type)):
                if task_type[i] == 'nc':
                    batch_size = 4
                    for j in range(batch_size):
                        current_emb = graph_emb[current_graph_id]
                        
                        # 1. Obtain initial features from each task-specific projector
                        nc_features = model.mm_projector_nc(current_emb)
                        lp_features = model.mm_projector_lp(current_emb)
                        
                        # 2. For the NC task, use NC features as queries and LP features as keys/values
                        # Pass in the task type so the proper padding mask is applied
                        enhanced_features = model.cross_task_attention(nc_features, lp_features, task_type='nc')
                        graph_feature = enhanced_features
                        
                        graph_features_list.append(graph_feature)
                        current_graph_id += 1
                        
                elif task_type[i] == 'lp':
                    batch_size = 2
                    for j in range(batch_size):
                        current_emb = graph_emb[current_graph_id]
                        
                        # 1. Obtain initial features from each task-specific projector
                        lp_features = model.mm_projector_lp(current_emb)
                        nc_features = model.mm_projector_nc(current_emb)
                        
                        # 2. For the LP task, use LP features as queries and NC features as keys/values
                        # LP tasks do not need a padding mask
                        enhanced_features = model.cross_task_attention(lp_features, nc_features, task_type='lp')

                        graph_feature = enhanced_features
                        
                        graph_features_list.append(graph_feature)
                        current_graph_id += 1
                else:
                    raise ValueError(f"Unknown task type: {task_type[i]}")
            model.node_attn_scores = node_attn_scores
            model.cross_task_info_flow = cross_task_info_flow
        

        # Default dual projector mode
        else:
            for i in range(len(task_type)):
                if task_type[i] == 'nc':
                    batch_size = 4
                    for j in range(batch_size):
                        graph_feature = model.mm_projector_nc(graph_emb[current_graph_id])
                        
                        graph_features_list.append(graph_feature)
                        current_graph_id += 1
                        
                elif task_type[i] == 'lp':
                    batch_size = 2
                    for j in range(batch_size):
                        graph_feature = model.mm_projector_lp(graph_emb[current_graph_id])
                        
                        graph_features_list.append(graph_feature)
                        current_graph_id += 1
                else:
                    raise ValueError(f"Unknown task type: {task_type[i]}")
        
        # Stack all graph features
        graph_features = torch.stack(graph_features_list, dim=0)
        
        # Apply graph mask if needed
        if graph is not None:
            graph_features[graph==DEFAULT_GRAPH_PAD_ID] = 0.
        
        return graph_features
    
    
    def encode_images(self, image_emb):
        image_features = self.get_model().mm_projector_image(image_emb)
        return image_features

    def encode_images_general(self, image_emb, task_type):
        image_features_list = []
        current_image_id = 0
        for i in range(len(task_type)):
            if task_type[i] == 'nc':
                for j in range(4):
                    image_feature = self.get_model().mm_projector_image_nc(image_emb[current_image_id])
                    image_features_list.append(image_feature)
                    current_image_id += 1
            elif task_type[i] == 'lp':
                for j in range(2):
                    image_feature = self.get_model().mm_projector_image_lp(image_emb[current_image_id])
                    image_features_list.append(image_feature)
                    current_image_id += 1
            else:
                raise ValueError(f"Unknown task type: {task_type[i]}")
        image_features = torch.stack(image_features_list, dim=0)
        return image_features
    
    def encode_texts(self, text_emb):
        text_features = self.get_model().mm_projector_text(text_emb)
        return text_features
    
    def encode_nodes(self, text_emb):
        text_features = self.get_model().mm_projector_node(text_emb)
        return text_features

    def inject_special_token(self, graph_emb):
        use_hop=self.config.use_hop
        sample_size = self.config.sample_neighbor_size
        assert graph_emb.shape[-2] == int((sample_size ** (use_hop + 1) - 1) / (sample_size - 1))
        assert self.model.special_token_emb.shape[0] == use_hop + 2
        new_graph_emb = []
        new_graph_emb.append(self.model.special_token_emb[0])
        cur=0
        for i in range(use_hop+1):
            cur_size = sample_size**i
            new_graph_emb.append(graph_emb[cur:cur+cur_size])
            cur+=cur_size
            new_graph_emb.append(self.model.special_token_emb[i+1])
        new_graph_emb = torch.concat(new_graph_emb, dim=0)
        return new_graph_emb

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, graph_emb=None, image_emb=None, text_emb=None, node_emb= None, graph=None, task_type=None
    ):  
        if past_key_values is not None and input_ids.shape[1] == 1:
            attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                        dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels
        # graph_features = self.encode_graphs(graphs, graph_emb)
        if graph_emb is not None:
            if getattr(self.config, 'is_cross_task_attention', False):
                graph_features = self.encode_graphs_general(graph_emb, graph, task_type)
            else:
                graph_features = self.encode_graphs(graph_emb, graph)
        if image_emb is not None:
            # if getattr(self.config, 'is_general_model', False):
            #     image_features = self.encode_images_general(image_emb, task_type)
            # else:
            image_features = self.encode_images(image_emb)
        if text_emb is not None:
            text_features = self.encode_texts(text_emb)
        if node_emb is not None:
            node_features = self.encode_nodes(node_emb)
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_graph_idx = 0
        cur_image_idx = 0
        cur_text_idx = 0
        cur_node_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == GRAPH_TOKEN_INDEX).sum() == 0 and (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0 and (cur_input_ids == TEXT_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_graph_features = graph_features[cur_graph_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_graph_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_graph_idx += 1
                continue
            # find all the GRAPH_TOKEN_INDEX and IMAGE_TOKEN_INDEX
            task = task_type[batch_idx]
            graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            text_token_indices = torch.where(cur_input_ids == TEXT_TOKEN_INDEX)[0]
            node_token_indices = torch.where(cur_input_ids == NODE_TOKEN_INDEX)[0]
            all_token_indices = torch.cat([graph_token_indices, image_token_indices, text_token_indices, node_token_indices])
            all_token_indices, token_types = torch.sort(all_token_indices)

            cur_new_input_embeds = []

            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            while all_token_indices.numel() > 0:
                token_index = all_token_indices[0]

                if token_index in graph_token_indices:
                    cur_features = graph_features[cur_graph_idx]
                    if getattr(self.config, 'is_data_task_general', False):
                        if task == 'nc':
                            half_len = cur_features.shape[0] // 2  
                            cur_features = cur_features[:half_len, :]  
                    cur_graph_idx += 1
                elif token_index in image_token_indices:
                    cur_features = image_features[cur_image_idx]
                    if getattr(self.config, 'is_data_task_general', False):
                        if task == 'nc':
                            half_len = cur_features.shape[0] // 2  
                            cur_features = cur_features[:half_len, :]  
                    cur_image_idx += 1
                elif token_index in text_token_indices:
                    cur_features = text_features[cur_text_idx]
                    cur_text_idx += 1
                elif token_index in node_token_indices:
                    cur_features = node_features[cur_node_idx]
                    cur_node_idx += 1
                
                if cur_features.dim() == 1:
                    cur_features = cur_features.unsqueeze(0)

                if hasattr(self.config, "mm_use_graph_special_token") and getattr(self.config, 'mm_use_graph_special_token', False):
                    cur_graph_features = self.inject_special_token(cur_graph_features)

                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:token_index]))
                cur_new_input_embeds.append(cur_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:token_index])
                    cur_new_labels.append(torch.full((cur_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[token_index+1:]
                cur_input_ids = cur_input_ids[token_index+1:]
                
                graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
                text_token_indices = torch.where(cur_input_ids == TEXT_TOKEN_INDEX)[0]
                node_token_indices = torch.where(cur_input_ids == NODE_TOKEN_INDEX)[0]

                all_token_indices = torch.cat([graph_token_indices, image_token_indices, text_token_indices, node_token_indices])
                all_token_indices, token_types = torch.sort(all_token_indices)

            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_graph_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)

            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels


    def initialize_graph_tokenizer(self, model_args, tokenizer):

        if model_args.mm_use_graph_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
