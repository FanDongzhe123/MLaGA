# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence
import pandas as pd

import torch
import pdb

import transformers
import itertools

import sys
sys.path.append(".")
sys.path.append("./utils")
sys.path.append("..")

from MMGIT.constants import IGNORE_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_IMAGE_TOKEN ,DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN, DEFAULT_GRAPH_PAD_ID, IMAGE_TOKEN_INDEX, GRAPH_TOKEN_INDEX
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler
import numpy as np
from graphllava_trainer import GraphLLaVATrainer
from MMGIT.model.builder import load_pretrained_model
from MMGIT.utils import disable_torch_init, tokenizer_graph_token, get_model_name_from_path

from MMGIT.model import *

import random
from tqdm import trange
import MMGIT.conversation as conversation_lib
import scipy.sparse as sp
import numpy as np


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_graph_start_end: bool = field(default=False)
    mm_use_graph_patch_token: bool = field(default=True)
    is_general_model: bool = field(default=False)
    is_data_task_general: bool = field(default=False)
    is_dual_projector: bool = field(default=False)
    is_shared_first_layer: bool = field(default=False)
    is_moe: bool = field(default=False)
    use_gate_network: bool = field(default=False)
    is_dual_projector_with_common: bool = field(default=False)
    is_moe_with_task_token: bool = field(default=False)
    is_soft_moe_with_task_token: bool = field(default=False)
    num_experts: int = field(default=8)
    top_k_experts: int = field(default=1)
    pretrained_lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained LoRA weights to initialize the model"}
    )
    pretrained_image_projector_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained image projector weights"}
    )
    pretrained_common_projector_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained common projector weights"}
    )
    is_cross_task_attention: bool = field(
        default=False,
        metadata={"help": "Use cross-task attention between NC and LP projectors"}
    )
    cross_attn_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads in cross-task attention module"}
    )
    attn_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for attention weights"}
    )
    ffn_dim: int = field(
        default=None,
        metadata={"help": "Dimension of FFN in cross-task attention, defaults to 4x hidden_dim"}
    )
    cross_attn_layers: int = field(
        default=1,
        metadata={"help": "Number of layers in cross-task attention module"}
    )


@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    pretrained_embedding_type: Optional[str] = field(default='sbert')
    use_hop: Optional[int] = field(default=2)
    sample_neighbor_size: Optional[int] = field(default=-1)
    use_task:Optional[str] = field(default="nc")
    use_dataset:Optional[str] = field(default="arxiv")
    template: Optional[str] = field(default="ND")
    use_neighbor: bool = False
    use_text_cls: bool = False
    dataset_task_mapping: Optional[str] = field(
        default=None,
        metadata={"help": "Mapping between datasets and tasks, format: 'dataset1:task1,dataset2:task2'"}
    )



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    dataloader_num_workers: Optional[int] = field(default=8)
    model_max_length: int = field(
        default=4096,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    group_by_modality_length: bool = field(default=False)
    data_task_generalization: bool = field(default=False)
    task_separate: bool = field(default=False)
    nc_first: bool = field(default=True)
    is_cold_start: bool = field(default=False)



def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    # if getattr(trainer.args, "tune_mm_mlp_adapter", False):
    #     # Only save Adapter
    #     keys_to_match = ['mm_projector']
    #     if getattr(trainer.args, "use_graph_start_end", False):
    #         keys_to_match.extend(['embed_tokens', 'embed_in'])

    #     weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    #     trainer.model.config.save_pretrained(output_dir)

    #     current_folder = output_dir.split('/')[-1]
    #     parent_folder = os.path.dirname(output_dir)
    #     if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
    #         if current_folder.startswith('checkpoint-'):
    #             mm_projector_folder = os.path.join(parent_folder, "mm_projector")
    #             os.makedirs(mm_projector_folder, exist_ok=True)
    #             torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
    #         else:
    #             torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
    #     return
    if getattr(trainer.args, "tune_mm_mlp_adapter", False) and getattr(trainer.args, "freeze_backbone", False):
        # Only save Adapter
        projector_list = ['mm_projector_graph', 'mm_projector_image', 'mm_projector_text', 'mm_projector_node', "mm_projector_nc", "mm_projector_lp", "mm_projector_shared_first_layer", "mm_projector_shared", "gate_network", "common_projector", "router", "expert_projectors", "cross_task_attention"]
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        for proj_key in projector_list:
            if not hasattr(trainer.model.get_model(), proj_key):
                continue
            keys_to_match = [proj_key]
            if getattr(trainer.args, "use_graph_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)

            if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
                if current_folder.startswith('checkpoint-'):
                    mm_projector_folder = os.path.join(parent_folder, f"{proj_key}")
                    os.makedirs(mm_projector_folder, exist_ok=True)
                    torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
                else:
                    torch.save(weight_to_save, os.path.join(output_dir, f'{proj_key}.bin'))
        if hasattr(trainer.model.get_model(), "task_token_embeddings"):
            task_tokens = {}
            for name, param in trainer.model.get_model().task_token_embeddings.named_parameters():
                task_tokens[name] = maybe_zero_3(param, ignore_status=True).cpu()
            
            if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
                if current_folder.startswith('checkpoint-'):
                    task_token_folder = os.path.join(parent_folder, "task_token_embeddings")
                    os.makedirs(task_token_folder, exist_ok=True)
                    torch.save(task_tokens, os.path.join(task_token_folder, f'{current_folder}.bin'))
                else:
                    torch.save(task_tokens, os.path.join(output_dir, 'task_token_embeddings.bin'))
        return

    elif getattr(trainer.args, "tune_mm_mlp_adapter", False) and not getattr(trainer.args, "freeze_backbone", False):
        projector_list = ['mm_projector_graph', 'mm_projector_image', 'mm_projector_text', 'mm_projector_node', "mm_projector_nc", "mm_projector_lp", "mm_projector_shared_first_layer", "mm_projector_shared", "gate_network", "common_projector", "router", "expert_projectors", "cross_task_attention"]
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        for proj_key in projector_list:
            if not hasattr(trainer.model.get_model(), proj_key):
                continue
            keys_to_match = [proj_key]
            if getattr(trainer.args, "use_graph_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)

            if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
                if current_folder.startswith('checkpoint-'):
                    mm_projector_folder = os.path.join(parent_folder, f"{proj_key}")
                    os.makedirs(mm_projector_folder, exist_ok=True)
                    torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
                else:
                    torch.save(weight_to_save, os.path.join(output_dir, f'{proj_key}.bin'))
        if hasattr(trainer.model.get_model(), "task_token_embeddings"):
            task_tokens = {}
            for name, param in trainer.model.get_model().task_token_embeddings.named_parameters():
                task_tokens[name] = maybe_zero_3(param, ignore_status=True).cpu()
            
            if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
                if current_folder.startswith('checkpoint-'):
                    task_token_folder = os.path.join(parent_folder, "task_token_embeddings")
                    os.makedirs(task_token_folder, exist_ok=True)
                    torch.save(task_tokens, os.path.join(task_token_folder, f'{current_folder}.bin'))
                else:
                    torch.save(task_tokens, os.path.join(output_dir, 'task_token_embeddings.bin'))
        
        if trainer.deepspeed:
            torch.cuda.synchronize()
            trainer.save_model(output_dir)
            return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation





def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_graph:
        input_ids = torch.stack([tokenizer_graph_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_graph:
                round_len = len(tokenizer_graph_token(rou, tokenizer))
                instruction_len = len(tokenizer_graph_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful multimodal graph reasoning assistant. You are able to understand visual content and graph-aware feature that the user provides, and assist the user with a variety of tasks using natural language."
    
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    tokenizer = copy.deepcopy(tokenizer)
    if has_graph:
        tokenizer.add_tokens(["<graph>"], special_tokens=True)
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    graph_token_index = tokenizer.convert_tokens_to_ids("<graph>")
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    chat_template = "{% for message in messages %}{{'<|begin_of_text|><|start_header_id|>' + message['role'] + '<|end_header_id|>' + '\n' + message['content'] + '<|eot_id|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    # unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids
    
    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            # if encode_id in unmask_tokens_idx:
            #     target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
            if encode_id == graph_token_index:
                input_id[idx] = GRAPH_TOKEN_INDEX

        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)


    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        # Conversation(system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.", roles=('USER', 'ASSISTANT'), messages=[['USER', 'Given a node-centered graph: <graph>, each node represents a paper, we need to classify the center node into 7 classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory, please tell me which class the center node belongs to?'], ['ASSISTANT', 'Rule_Learning']], offset=0, sep_style=<SeparatorStyle.TWO: 2>, sep=' ', sep2='</s>', version='v1', skip_next=False)
        conversations.append(conv.get_prompt())
        #["A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. : Given a node-centered multimodal graph: <graph>, each node contains a text description and a figure. Each node represents products sold in Amazon, and edges between products indicate they are purchased together. we need to classify the center node into 19 classes: Movies, Genre for Featured Categories, Studio Specials, Musicals & Performing Arts, A&E Home Video, TV, Science Fiction & Fantasy, Boxed Sets, Walt Disney Studios Home Entertainment, Paramount Home Entertainment, Blu-ray, Art House & International, Criterion Collection, Holidays & Seasonal, Music Artists, BBC, Fully Loaded DVDs, Independently Distributed, HBO, Classics, please tell me which class the center node belongs to? : Genre for Featured Categories</s>"]

    # Tokenize conversations

    if has_graph:
        input_ids = torch.stack([tokenizer_graph_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        # [1, 212]
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        # total_len = int(target.ne(tokenizer.pad_token_id).sum())
        total_len = target.shape[0]

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep # part before GPT value + ASSISTANT:

            if has_graph:
                round_len = len(tokenizer_graph_token(rou, tokenizer))
                instruction_len = len(tokenizer_graph_token(parts[0], tokenizer)) - 2 #remove start and end token
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX #mask the part before GPT value

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_graph_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_graph_token(rou, tokenizer)) + len(tokenizer_graph_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_graph_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )



def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_graph=has_graph)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_graph=has_graph)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    if conversation_lib.default_conversation.version == "llama_v3":
        return preprocess_llama3(sources, tokenizer, has_graph=has_graph)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_graph_token(prompt, tokenizer)) for prompt in prompts]

    if has_graph:
        input_ids = [tokenizer_graph_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_graph:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)
    
class LazySupervisedGraphDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(LazySupervisedGraphDataset, self).__init__()
        self.use_dataset = data_args.use_dataset.split('-')
        self.use_hop = data_args.use_hop
        self.template = data_args.template
        self.datas = {}
        list_data_dict = []
        self.pretrained_graph_embs = {}
        self.pretrained_image_embs = {}
        self.pretrained_text_embs = {}
        self.pretrained_node_embs = {}
        self.neighbor_lists = {}
        self.use_neighbor = False #if use node sequence in the template
        self.use_text_cls = False #if use text cls token in the template
        self.use_all = False #if use text/image/qquery/
        self.graph_only = False

        # Parse the dataset_task_mapping argument
        dataset_to_tasks = {}
        if data_args.dataset_task_mapping:
            print(f"Using dataset-specific tasks mapping: {data_args.dataset_task_mapping}")
            for mapping in data_args.dataset_task_mapping.split(','):
                parts = mapping.strip().split(':')
                if len(parts) == 2:
                    dataset, tasks = parts
                    dataset_to_tasks[dataset] = tasks.split('-')
                    print(f"  {dataset} will use tasks: {tasks}")
        
        # Default to the global task setting
        self.use_task = data_args.use_task.split('-')
        
        for d, dataset in enumerate(self.use_dataset):
            repeat = 1
            if "." in dataset:
                # cora: use cora dataset
                # cora.3: repeat cora 3 times
                ds = dataset.split('.')
                repeat = int(ds[1])
                dataset = ds[0]
                
            # Resolve the data path
            if "Movies" in dataset:
                data_path = "dataset/Movies/"
            elif "Toys" in dataset:
                data_path = "dataset/Toys/"
            elif "Grocery" in dataset:
                data_path = "dataset/Grocery/"
            elif "Health" in dataset:
                data_path = "dataset/Health/"
            elif "Beauty" in dataset:
                data_path = "dataset/Beauty/"
            elif "VideoGames" in dataset:
                data_path = "dataset/VideoGames/"
            elif "CD" in dataset:
                data_path = "dataset/CD/"
            elif "Arts" in dataset:
                data_path = "dataset/Arts/"
            elif "Automotive" in dataset:
                data_path = "dataset/Automotive/"
            elif "RedditS" in dataset:
                data_path = "dataset/RedditS/"
            elif "Goodreads" in dataset:
                data_path = "/vast/df2362/Goodreads/"
            elif "Art500K" in dataset:
                data_path = "dataset/Art500K/"
            else:
                print(f"{dataset} not exists!!!!")
                raise ValueError
                
            # Load the pretrained embeddings
            data_dir = os.path.dirname(data_path)
            pretrained_graph_emb, pretrained_image_emb, pretrained_text_emb = self.load_pretrain_embedding_graph(data_dir, data_args.pretrained_embedding_type)
            
            self.pretrained_graph_embs[dataset] = pretrained_graph_emb
            self.pretrained_image_embs[dataset] = pretrained_image_emb
            self.pretrained_text_embs[dataset] = pretrained_text_emb

            # Decide which tasks should be used for the current dataset
            current_tasks = dataset_to_tasks.get(dataset, self.use_task)
            print(f"Dataset {dataset} will load tasks: {current_tasks}")
            
            # Load data for every task that applies to this dataset
            for task in current_tasks:
                task_list_data_dict = []
                if task == "nc":
                    if data_args.template == "TIQ":
                        data_path = os.path.join(data_dir, f"{dataset}_train_TIQ.jsonl")
                    elif data_args.template == "TIQ_demo":
                        data_path = os.path.join(data_dir, f"{dataset}_train_TIQ_demo.jsonl")
                    elif data_args.template == "HO":
                        data_path = os.path.join(data_dir, f"sampled_2_10_train.jsonl")
                    elif data_args.template == "ND":
                        data_path = os.path.join(data_dir, f"{dataset}_train_ND.jsonl")
                        data_args.use_neighbor = True
                        self.use_neighbor = True
                        self.graph_only = True
                    else:
                        data_path = os.path.join(data_dir, f"sampled_{data_args.use_hop}_{data_args.sample_neighbor_size}_train.jsonl")
                        
                    # Handle special template cases
                    if self.use_all:
                        self.pretrained_node_embs[dataset] = pretrained_graph_emb
                    if data_args.use_neighbor or self.use_all:
                        self.structure_emb = torch.load(
                                f"dataset/laplacian_{data_args.use_hop}_{data_args.sample_neighbor_size}.pt")
                        if data_args.pretrained_embedding_type == 'clip_all' or data_args.pretrained_embedding_type == 'clip_qquery':
                            pretrained_graph_emb = torch.mean(pretrained_graph_emb, dim=1)
                        if data_args.template == "ND":
                            if data_args.pretrained_embedding_type == 'all':
                                pretrained_graph_emb = torch.mean(pretrained_graph_emb, dim=1)
                        if data_args.pretrained_embedding_type == 'clip_all':
                            pretrained_graph_emb = torch.cat([pretrained_graph_emb, pretrained_text_emb, pretrained_image_emb], dim=1)
                        self.pretrained_graph_embs[dataset] = pretrained_graph_emb
                        
                    # Load the data file
                    if os.path.exists(data_path):
                        with open(data_path, 'r') as file:
                            for line in file:
                                l = json.loads(line)
                                l["dataset"] = dataset
                                l["task_type"] = "nc"  # set task type to nc
                                task_list_data_dict.append(l)
                    else:
                        raise ValueError(f"Data path {data_path} not found")
                        
                elif task == "lp":
                    if data_args.template == "TIQ_s":
                        data_path = os.path.join(data_dir, f"{dataset}_train_TIQ_lp.jsonl")
                    elif data_args.template == "TIQ":
                        data_path = os.path.join(data_dir, f"{dataset}_train_TIQ_lp.jsonl")
                    elif data_args.template == "TIQ_demo":
                        data_path = os.path.join(data_dir, f"{dataset}_train_TIQ_lp_demo.jsonl")

                    # Load the data file
                    if os.path.exists(data_path):
                        with open(data_path, 'r') as file:
                            for line in file:
                                l = json.loads(line)
                                l["dataset"] = dataset
                                l["task_type"] = "lp"  # set task type to lp
                                task_list_data_dict.append(l)
                    else:
                        raise ValueError(f"Data path {data_path} not found")
                else:
                    print(f"{task} not exist!!!")
                    raise ValueError

                # Handle data repetition
                if repeat > 1:
                    base_task_list_data_dict = copy.copy(task_list_data_dict)
                    for _ in range(repeat-1):
                        task_list_data_dict += base_task_list_data_dict
                
                rank0_print(f"Dataset {dataset} Task {task}, size {len(task_list_data_dict)}")
                list_data_dict.extend(task_list_data_dict)

        # Shuffle the data randomly
        random.shuffle(list_data_dict)
        rank0_print(f"Formatting inputs...Skip in lazy mode, size {len(list_data_dict)}")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def load_pretrain_embedding_graph(self, data_dir, pretrained_embedding_type):
        if pretrained_embedding_type == "clip":
            graph_emb = torch.load(os.path.join(data_dir, "clip.pt"))
            image_emb = torch.load(os.path.join(data_dir, "clip_image_new.pt"))
            text_emb = torch.load(os.path.join(data_dir, "clip_text_new.pt"))
            if graph_emb.is_cuda:
                graph_emb = graph_emb.to("cpu")
            if image_emb.is_cuda:
                image_emb = image_emb.to("cpu")
            if text_emb.is_cuda:
                text_emb = text_emb.to("cpu")
            return graph_emb, image_emb, text_emb
        elif pretrained_embedding_type == "clip_mix":
            graph_emb = torch.load(os.path.join(data_dir, "clip_mix.pt"))
            image_emb = torch.load(os.path.join(data_dir, "clip_image_new.pt"))
            text_emb = torch.load(os.path.join(data_dir, "clip_text_new.pt"))
            if graph_emb.is_cuda:
                graph_emb = graph_emb.to("cpu")
            if image_emb.is_cuda:
                image_emb = image_emb.to("cpu")
            if text_emb.is_cuda:
                text_emb = text_emb.to("cpu")
            return graph_emb, image_emb, text_emb
        elif pretrained_embedding_type == "clip_mix_new":
            graph_emb = torch.load(os.path.join(data_dir, "clip_mix_new.pt"))
            image_emb = torch.load(os.path.join(data_dir, "clip_image_new.pt"))
            text_emb = torch.load(os.path.join(data_dir, "clip_text_new.pt"))
            if graph_emb.is_cuda:
                graph_emb = graph_emb.to("cpu")
            if image_emb.is_cuda:
                image_emb = image_emb.to("cpu")
            if text_emb.is_cuda:
                text_emb = text_emb.to("cpu")
            return graph_emb, image_emb, text_emb
        elif pretrained_embedding_type == "clip_sbert":
            graph_emb = torch.load(os.path.join(data_dir, "clip_sbert.pt"))
            image_emb = torch.load(os.path.join(data_dir, "clip_image_new.pt"))
            text_emb = torch.load(os.path.join(data_dir, "clip_text_new.pt"))
            if graph_emb.is_cuda:
                graph_emb = graph_emb.to("cpu")
            if image_emb.is_cuda:
                image_emb = image_emb.to("cpu")
            if text_emb.is_cuda:
                text_emb = text_emb.to("cpu")
            return graph_emb, image_emb, text_emb
        elif pretrained_embedding_type == "clip_sbert_blip":
            graph_emb = torch.load(os.path.join(data_dir, "clip_sbert_blip.pt"))
            image_emb = torch.load(os.path.join(data_dir, "clip_image_new.pt"))
            text_emb = torch.load(os.path.join(data_dir, "clip_text_new.pt"))
            if graph_emb.is_cuda:
                graph_emb = graph_emb.to("cpu")
            if image_emb.is_cuda:
                image_emb = image_emb.to("cpu")
            if text_emb.is_cuda:
                text_emb = text_emb.to("cpu")
            return graph_emb, image_emb, text_emb
        elif pretrained_embedding_type == 'clip_text':
            graph_emb = torch.load(os.path.join(data_dir, "clip_text_new.pt"))
            return graph_emb, None, None
        elif pretrained_embedding_type == 'all':
            graph_emb = torch.load(os.path.join(data_dir, "query_token_all.pt"))
            if graph_emb.is_cuda:
                graph_emb = graph_emb.to("cpu")
            return graph_emb, None, None
        elif pretrained_embedding_type == 'all_cat':
            graph_emb = torch.load(os.path.join(data_dir, "query_token_all.pt"))
            if graph_emb.is_cuda:
                graph_emb = graph_emb.to("cpu")
            graph_emb = graph_emb.view(graph_emb.shape[0], -1)
            return graph_emb, None, None
        elif pretrained_embedding_type == 'clip_image':
            graph_emb = torch.load(os.path.join(data_dir, "clip_image_new.pt"))
            return graph_emb, None, None
        elif pretrained_embedding_type == 'clip_image_text':
            image_emb = torch.load(os.path.join(data_dir, "clip_image_new.pt"))
            text_emb = torch.load(os.path.join(data_dir, "clip_text_new.pt"))
            graph_emb = torch.cat([text_emb, image_emb], dim=1)
            return graph_emb, None, None
        elif pretrained_embedding_type == 'clip_qquery':
            graph_emb = torch.load(os.path.join(data_dir, "clip.pt"))
            if graph_emb.is_cuda:
                graph_emb = graph_emb.to("cpu")
            return graph_emb, None, None
        elif pretrained_embedding_type == 'clip_all':
            graph_emb = torch.load(os.path.join(data_dir, f"clip.pt"))
            image_emb = torch.load(os.path.join(data_dir, "clip_image_new.pt"))
            text_emb = torch.load(os.path.join(data_dir, "clip_text_new.pt"))
            if graph_emb.is_cuda:
                graph_emb = graph_emb.to("cpu")
            return graph_emb, image_emb, text_emb
    
    def load_neighbors(self, data_dir):
        neighbor_list = torch.load(os.path.join(data_dir, "neighbors.pt"))
        return neighbor_list

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            graph_token_size = len(sample['graphs']) if 'graphs' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + graph_token_size)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'graph' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_graph=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        # image exist in the data
        if self.use_neighbor or self.use_all:
            if self.use_all:
                id = self.list_data_dict[i]["id"]
                data_dict['node_emb'] = self.pretrained_node_embs[self.list_data_dict[i]["dataset"]][id].unsqueeze(0)
            if not isinstance(self.list_data_dict[i]['graph'][0], list):
                self.list_data_dict[i]['graph'] = [self.list_data_dict[i]['graph']]
            if self.template == "ND_sequence" or self.template == "ND_all" or self.template == "ND":
                graph = torch.LongTensor(self.list_data_dict[i]['graph'])
                mask = graph != DEFAULT_GRAPH_PAD_ID
                masked_graph_emb = self.pretrained_graph_embs[self.list_data_dict[i]["dataset"]][graph[mask]]
                s, n, d = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
                graph_emb = torch.zeros((s, n, d))
                graph_emb[mask] = masked_graph_emb
                if self.structure_emb is not None:
                    graph_emb = torch.cat([graph_emb, self.structure_emb.unsqueeze(0).expand(s, -1, -1)], dim=-1)
                data_dict['graph'] = graph
                data_dict['graph_emb'] = graph_emb
        else:
            if self.template == "TIQ_demo" or self.template == "TIQ_demo_image_only" or self.template == "TIQ_demo_1" or self.template == "TIQ_demo_2" or self.template == "TIQ_demo_text_only":
                demo_id = self.list_data_dict[i]["demo_id"]
                id = self.list_data_dict[i]["id"]
                id = [id]
                all_graph_id = id + demo_id
                graph_emb_list = []
                for idx in all_graph_id:
                    if isinstance(idx, list):
                        pair_emb_list = []
                        for pair_id in idx:
                            pair_emb_list.append(self.pretrained_graph_embs[self.list_data_dict[i]["dataset"]][pair_id])
                        cat_pair_emb = torch.cat(pair_emb_list, dim=0)
                        graph_emb_list.append(cat_pair_emb)
                    else:
                        pair_emb_list = []
                        node_emb = self.pretrained_graph_embs[self.list_data_dict[i]["dataset"]][idx]
                        pair_emb_list.append(node_emb)
                        padding_emb = torch.zeros_like(node_emb)
                        pair_emb_list.append(padding_emb)
                        cat_pair_emb = torch.cat(pair_emb_list, dim=0)
                        graph_emb_list.append(cat_pair_emb)
                data_dict['graph_emb'] = torch.stack(graph_emb_list, dim=0)

            else:
                if isinstance(self.list_data_dict[i]["id"], list):
                    graph_emb_list = []
                    for idx in self.list_data_dict[i]["id"]:
                        graph_emb_list.append(self.pretrained_graph_embs[self.list_data_dict[i]["dataset"]][idx])
                    if self.template == "TIQ_cat":
                        data_dict['graph_emb'] = torch.cat(graph_emb_list, dim=0).unsqueeze(0)
                    else:
                        data_dict['graph_emb'] = torch.stack(graph_emb_list, dim=0)
                else:
                    id = self.list_data_dict[i]["id"]
                    data_dict['graph_emb'] = self.pretrained_graph_embs[self.list_data_dict[i]["dataset"]][id].unsqueeze(0)
                    
        if not self.graph_only:
            if self.template == "TIQ_demo" or self.template == "TIQ_demo_image_only" or self.template == "TIQ_demo_1" or self.template == "TIQ_demo_2" or self.template == "TIQ_demo_text_only":
                demo_id = self.list_data_dict[i]["demo_id"]
                id = self.list_data_dict[i]["id"]
                id = [id]
                all_image_id = id + demo_id
                image_emb_list = []
                for idx in all_image_id:
                    if isinstance(idx, list):
                        pair_emb_list = []
                        for pair_id in idx:
                            pair_emb_list.append(self.pretrained_image_embs[self.list_data_dict[i]["dataset"]][pair_id])
                        cat_pair_emb = torch.stack(pair_emb_list, dim=0)
                        image_emb_list.append(cat_pair_emb)
                    else:
                        pair_emb_list = []
                        image_emb = self.pretrained_image_embs[self.list_data_dict[i]["dataset"]][idx]
                        pair_emb_list.append(image_emb)
                        padding_emb = torch.zeros_like(image_emb)
                        pair_emb_list.append(padding_emb)
                        cat_pair_emb = torch.stack(pair_emb_list, dim=0)
                        image_emb_list.append(cat_pair_emb)
                data_dict['image_emb'] = torch.stack(image_emb_list, dim=0)
            else:
                image_emb_list = []
                if isinstance(self.list_data_dict[i]["id"], list):
                    for idx in self.list_data_dict[i]["id"]:
                        image_emb_list.append(self.pretrained_image_embs[self.list_data_dict[i]["dataset"]][idx])
                    if self.template == "TIQ_cat":
                        data_dict['image_emb'] = torch.stack(image_emb_list, dim=0).unsqueeze(0)
                    else:
                        data_dict['image_emb'] = torch.stack(image_emb_list, dim=0)
                else:
                    id = self.list_data_dict[i]["id"]
                    data_dict['image_emb'] = self.pretrained_image_embs[self.list_data_dict[i]["dataset"]][id].unsqueeze(0)
                    
        if self.use_text_cls:
            data_dict['text_emb'] = self.pretrained_text_embs[self.list_data_dict[i]["dataset"]][id].unsqueeze(0)
            
        data_dict['task_type'] = self.list_data_dict[i]["task_type"]

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        task_types = [instance["task_type"] for instance in instances]
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'graph' in instances[0]:
            graph = [instance['graph'] for instance in instances]
            graph_emb = [instance['graph_emb'] for instance in instances]
            batch['graph'] = torch.cat(graph, dim=0)
            batch['graph_emb'] = torch.cat(graph_emb, dim=0)
        else:
            graph_emb = [instance['graph_emb'] for instance in instances]
            batch['graph_emb'] = torch.cat(graph_emb, dim=0)
        if 'node_emb' in instances[0]:
            node_emb = [instance['node_emb'] for instance in instances]
            batch['node_emb'] = torch.cat(node_emb, dim=0)
        if 'text_emb' in instances[0]:
            text_emb = [instance['text_emb'] for instance in instances]
            batch['text_emb'] = torch.cat(text_emb, dim=0)
        if 'image_emb' in instances[0]:
            image_emb = [instance['image_emb'] for instance in instances]
            batch['image_emb'] = torch.cat(image_emb, dim=0)
        batch['task_type'] = task_types

        return batch
    
class BalancedTaskSampler(Sampler):
    """Sampler that ensures each batch contains the same number of samples per task."""
    
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group data by task type
        self.task_indices = {}
        for i, sample in enumerate(dataset.list_data_dict):
            task_type = sample["task_type"]
            if task_type not in self.task_indices:
                self.task_indices[task_type] = []
            self.task_indices[task_type].append(i)
        
        self.task_types = list(self.task_indices.keys())
        print(f"Task types: {self.task_types}")
        print(f"Task data counts: {[len(self.task_indices[t]) for t in self.task_types]}")
        
        # Number of samples per task in a batch
        self.samples_per_task = batch_size // len(self.task_types)
        
        # Handle the case when the batch size is not divisible by the number of tasks
        self.remainder = batch_size % len(self.task_types)
        
        # Compute how many complete batches can be formed
        min_task_size = min([len(indices) for indices in self.task_indices.values()])
        max_complete_batches = min_task_size // self.samples_per_task
        
        # Total size across all final batches
        self.total_size = max_complete_batches * batch_size

        if not drop_last and min_task_size % self.samples_per_task != 0:
            # If there is a remainder and we do not drop the last incomplete batch
            self.total_size += batch_size
        
        print(f"Total batches: {self.total_size // batch_size}")
    
    def __iter__(self):
        # Shuffle the indices for each task
        task_indices = {}
        for task_type in self.task_types:
            indices = self.task_indices[task_type].copy()
            np.random.shuffle(indices)
            task_indices[task_type] = indices
        
        # Per-task sample counts within a batch
        samples_per_task = {t: self.samples_per_task for t in self.task_types}
        
        # Handle the remainder (e.g. if batch size is 7 with 3 tasks, the first `remainder` tasks each get 1 extra sample)
        for i, task_type in enumerate(self.task_types[:self.remainder]):
            samples_per_task[task_type] += 1
        
        # Build batches
        batches = []
        current_indices = {t: 0 for t in self.task_types}
        
        while True:
            # Check whether all tasks still have enough samples to form a batch
            can_form_batch = True
            for task_type in self.task_types:
                if current_indices[task_type] + samples_per_task[task_type] > len(task_indices[task_type]):
                    can_form_batch = False
                    break
            
            if not can_form_batch:
                break
            
            # Pick the required number of samples from each task
            batch = []
            for task_type in self.task_types:
                start = current_indices[task_type]
                end = start + samples_per_task[task_type]
                batch.extend(task_indices[task_type][start:end])
                current_indices[task_type] = end
            
            # Shuffle samples inside the batch
            np.random.shuffle(batch)
            batches.extend(batch)
        
        return iter(batches)
    
    def __len__(self):
        return self.total_size

def make_balanced_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Create a data module with balanced tasks."""
    train_dataset = LazySupervisedGraphDataset(tokenizer=tokenizer,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator,
                sampler_cls=BalancedTaskSampler) 


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedGraphDataset(tokenizer=tokenizer,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def _train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    # if training_args.data_task_generalization:
    #     training_args.output_dir = training_args.output_dir + "_general"
    #     print(f"update output dir:{training_args.output_dir}")

    if "tmp" not in training_args.output_dir and os.path.exists(training_args.output_dir):
        if bool(os.listdir(training_args.output_dir)):
            print(f"{training_args.output_dir} already exists and not empty!!!!")
            return
        print(f"{training_args.output_dir} already exists!!!!")

    if data_args.pretrained_embedding_type in ['sbert', 'simteg_sbert']:
        model_args.mm_hidden_size = 384
    elif data_args.pretrained_embedding_type in ["simteg_e5", "simteg_roberta", "roberta"]:
        model_args.mm_hidden_size = 1024
    elif data_args.pretrained_embedding_type in ["simteg"]:
        model_args.mm_hidden_size = 1024*2+384
    elif data_args.pretrained_embedding_type in ["clip", "clip_sbert", "clip_sbert_blip", "clip_mix", "clip_mix_new"]:
        model_args.mm_hidden_size = 1024
        model_args.mm_image_hidden_size = 768
    elif data_args.pretrained_embedding_type in ["all"]:
        model_args.mm_hidden_size = 1024
    elif data_args.pretrained_embedding_type in ["all_cat"]:
        model_args.mm_hidden_size = 1024*32
        # model_args.mm_node_hidden_size = 1024
    elif data_args.pretrained_embedding_type in ["clip_all"]:
        model_args.mm_hidden_size = 1024 + 768*2
    elif data_args.pretrained_embedding_type in ["clip_image_text"]:
        model_args.mm_hidden_size = 768*2
    elif data_args.pretrained_embedding_type in ["clip_text", "clip_image"]:
        model_args.mm_hidden_size = 768
    elif data_args.pretrained_embedding_type in ["clip_qquery"]:
        model_args.mm_hidden_size = 1024

    else:
        raise ValueError
    
    if 'sequence' in data_args.template or 'all' in data_args.template or data_args.template == 'ND':
        data_args.use_neighbor = True
    elif 'text_cls' in data_args.template:
        data_args.use_text_cls = True

    if data_args.use_neighbor:
        data_args.structure_embedding_dim = int((data_args.sample_neighbor_size ** (data_args.use_hop + 1) - 1) / (data_args.sample_neighbor_size - 1))
        model_args.mm_hidden_size += data_args.structure_embedding_dim
    # if data_args.template == "ND":
    #     data_args.structure_embedding_dim = int((data_args.sample_neighbor_size ** (data_args.use_hop + 1) - 1) / (data_args.sample_neighbor_size - 1))
        # model_args.mm_hidden_size += data_args.structure_embedding_dim
    print(f"mm_hidden_size: {model_args.mm_hidden_size}")
    

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    use_pretrained_lora = model_args.pretrained_lora_path is not None

    if 'mpt' in model_args.model_name_or_path:
        # config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        # config.attn_config['attn_impl'] = training_args.mpt_attn_impl
        # model = LlagaMPTForCausalLM.from_pretrained(
        #     model_args.model_name_or_path,
        #     config=config,
        #     cache_dir=training_args.cache_dir,
        #     **bnb_model_from_pretrained_args
        # )
        pass
    elif 'opt' in model_args.model_name_or_path:
        # model = LlagaOPTForCausalLM.from_pretrained(
        #     model_args.model_name_or_path,
        #     cache_dir=training_args.cache_dir,
        #     **bnb_model_from_pretrained_args
        # )
        pass
    elif use_pretrained_lora:
        rank0_print(f"Loading pretrained LoRA weights from {model_args.pretrained_lora_path}")
        model = GraphLLaVALlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(
            model, 
            model_args.pretrained_lora_path,
        )
        print('Merging LoRA weights...')
        model = model.merge_and_unload()


    else:
        model = GraphLLaVALlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )

    model.config.use_cache = False

    # if model_args.freeze_backbone:
    #     model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    elif 'opt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            # model_max_length = 4096
            model_max_length=training_args.model_max_length
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # if model_args.vision_tower is not None:
    model.get_model().initialize_graph_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )

    data_args.is_multimodal = True
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        # model.requires_grad_(False)
        if not training_args.lora_enable:
            if model_args.freeze_backbone:
                rank0_print("Freezing backbone model, only training projectors...")
                model.requires_grad_(False)
            else:
                rank0_print("Training both backbone LLM and projectors...")
        if hasattr(model.get_model(), "mm_projector_graph"):
            for p in model.get_model().mm_projector_graph.parameters():
                p.requires_grad = True
        if hasattr(model.get_model(), "mm_projector_text"):
            for p in model.get_model().mm_projector_text.parameters():
                p.requires_grad = True
        if hasattr(model.get_model(), "mm_projector_image"):
            for p in model.get_model().mm_projector_image.parameters():
                p.requires_grad = True
        if hasattr(model.get_model(), "mm_projector_node"):
            for p in model.get_model().mm_projector_node.parameters():
                p.requires_grad = True
        if hasattr(model.get_model(), "common_projector"):
            for p in model.get_model().common_projector.parameters():
                p.requires_grad = True
        if hasattr(model.get_model(), "mm_projector_lp"):
            for p in model.get_model().mm_projector_lp.parameters():
                p.requires_grad = True
        if hasattr(model.get_model(), "mm_projector_nc"):
            for p in model.get_model().mm_projector_nc.parameters():
                p.requires_grad = True
        if hasattr(model.get_model(), "task_token_embeddings"):
            for name, param in model.get_model().task_token_embeddings.named_parameters():
                param.requires_grad = True
        if hasattr(model.get_model(), "router"):
            for p in model.get_model().router.parameters():
                p.requires_grad = True
        if hasattr(model.get_model(), "expert_projectors"):
            for p in model.get_model().expert_projectors.parameters():
                p.requires_grad = True
        if hasattr(model.get_model(), "cross_task_attention"):
            for p in model.get_model().cross_task_attention.parameters():
                p.requires_grad = True

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    print(f"Trainable parameters:")
    print(trainable_names)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_graph_start_end = data_args.mm_use_graph_start_end = model_args.mm_use_graph_start_end
    training_args.mm_use_graph_start_end = model_args.mm_use_graph_start_end
    model.initialize_graph_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # if training_args.data_task_generalization:
    #     model_path = './checkpoints/Movies-Toys-Arts-VideoGames/graphllava-vicuna-7b-clip_mix-2-10-2-layer-mlp-TIQ_demo-projector_nc_2'
    #     print(f"Loaded from {model_path}. Model Base: {model_args.model_name_or_path}")
    #     model_name = get_model_name_from_path(model_path)
    #     tokenizer, model, context_len = load_pretrained_model(model_path, model_args.model_name_or_path, model_name,
    #                                                       cache_dir=training_args.cache_dir)
    if training_args.task_separate:
        if training_args.nc_first:
            print("Training NC and LP separately")
            print(f"Training NC and LP separately, NC first")
            data_args.use_task = "nc"
            data_module_nc = make_supervised_data_module(tokenizer, data_args)
            trainer = GraphLLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module_nc)
            trainer.train()
            print(f"Training NC and LP separately, LP next")
            data_args.use_task = "lp"
            data_module_lp = make_supervised_data_module(tokenizer, data_args)
            trainer.train_dataset = data_module_lp["train_dataset"]
            trainer.data_collator = data_module_lp["data_collator"]
            trainer.train()
        else:
            print("Training NC and LP separately")
            print(f"Training NC and LP separately, LP first")
            data_args.use_task = "lp"
            data_module_nc = make_supervised_data_module(tokenizer, data_args)
            trainer = GraphLLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module_nc)
            trainer.train()
            print(f"Training NC and LP separately, NC next")
            data_args.use_task = "nc"
            data_module_lp = make_supervised_data_module(tokenizer, data_args)
            trainer.train_dataset = data_module_lp["train_dataset"]
            trainer.data_collator = data_module_lp["data_collator"]
            trainer.train()
    elif model_args.is_dual_projector:
        # data_args.use_task = "nc"
        # nc_mod = make_supervised_data_module(tokenizer, data_args)
        # data_args.use_task = "lp"
        # lp_mod = make_supervised_data_module(tokenizer, data_args)
        # trainer = DualProjectorTrainer(
        # model=model,
        # tokenizer=tokenizer,
        # args=training_args,
        # train_dataset1=nc_mod['train_dataset'],
        # train_dataset2=lp_mod['train_dataset'],
        # data_collator=nc_mod['data_collator'],
        # )
        # trainer.train()
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        
        # Create the trainer with balanced tasks enabled
        trainer = GraphLLaVATrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            balance_tasks=True,  # enable task balancing
            **data_module
        )
        trainer.train()
    elif training_args.is_cold_start:
        print("=== Cold Start Phase ===")

        data_args.use_task = "nc"
        data_module_nc = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

        trainer = GraphLLaVATrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            **data_module_nc
        )

        # Save the total epoch count and switch to 1-epoch training
        total_epochs = training_args.num_train_epochs
        trainer.args.num_train_epochs = 1
        trainer.args.save_strategy = "no"  # do not save checkpoints during the cold start
        trainer.train()

        print("=== Cold Start Finished ===")

        # === Step 2: Continue training phase ===
        print("=== Continue Fine-tuning Phase ===")
        data_args.use_task = "nc-lp"
        data_module_all = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

        # Swap in the new data and update the trainer's dataset and collator
        trainer.train_dataset = data_module_all["train_dataset"]
        trainer.data_collator = data_module_all["data_collator"]
        trainer.args.num_train_epochs = total_epochs  # restore the original epoch count
        trainer.args.save_strategy = training_args.save_strategy  # restore the save strategy

        trainer.train(resume_from_checkpoint=False)

    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)


        trainer = GraphLLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    balance_tasks=True,
                    **data_module)

        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    random.seed(0)
    _train()