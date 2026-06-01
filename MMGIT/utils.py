import torch
from .constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, TEXT_TOKEN_INDEX, NODE_TOKEN_INDEX ,DEFAULT_TEXT_TOKEN,IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_NODE_TOKEN
import re
import pdb

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-') or model_paths[-1].startswith("epoch-"):
        # return model_paths[-2] + "_" + model_paths[-1]
        return model_paths[-2]
    else:
        return model_paths[-1]
'''
Tokenize prompt=> split by <graph>
'''
def tokenizer_graph_token(prompt, tokenizer, text_token_index=TEXT_TOKEN_INDEX, image_token_index=IMAGE_TOKEN_INDEX, graph_token_index=GRAPH_TOKEN_INDEX, node_token_index= NODE_TOKEN_INDEX, return_tensors=None): 
    # prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_GRAPH_TOKEN)]
    # def insert_separator(X, sep):
    #     return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
    pattern = f"({DEFAULT_GRAPH_TOKEN}|{DEFAULT_IMAGE_TOKEN}|{DEFAULT_TEXT_TOKEN}|{DEFAULT_NODE_TOKEN})"
    chunks = re.split(pattern, prompt) #split by all the sprcial tokens
    input_ids = []
    offset = 0
    # if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
    #     offset = 1
    #     input_ids.append(prompt_chunks[0][0])
    #judge if the first chunk start with bos_token
    if chunks and chunks[0]:
        first_chunk_tokenized = tokenizer(chunks[0]).input_ids
        if first_chunk_tokenized and first_chunk_tokenized[0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(first_chunk_tokenized[0])
            input_ids.extend(first_chunk_tokenized[offset:])
        else:
            input_ids.extend(first_chunk_tokenized)
    else:
        pass

    for seg in chunks[1:]:
        if seg == DEFAULT_GRAPH_TOKEN:
            input_ids.append(graph_token_index)

        elif seg == DEFAULT_IMAGE_TOKEN:
            input_ids.append(image_token_index)

        elif seg == DEFAULT_TEXT_TOKEN:
            input_ids.append(text_token_index)
        elif seg == DEFAULT_NODE_TOKEN:
            input_ids.append(node_token_index)

        elif seg:
            tokenized = tokenizer(seg).input_ids
            input_ids.extend(tokenized)
        else:
            pass


    # for x in insert_separator(prompt_chunks, [graph_token_index] * (offset + 1)):
    #     input_ids.extend(x[offset:])
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)