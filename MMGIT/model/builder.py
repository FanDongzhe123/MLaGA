#    Copyright 2023 Haotian Liu
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
import warnings
import sys
sys.path.append("..")
sys.path.append("./language_model")
sys.path.append(".")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
import pdb


from .language_model.GraphLLaVA_llama import GraphLLaVALlamaForCausalLM
from MMGIT.constants import DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN
from huggingface_hub import hf_hub_download




def load_pretrained_model(model_path, model_base, model_name, pretrained_lora_path ,load_8bit=False, load_4bit=False, device_map="auto", device="cuda", cache_dir="../../checkpoint"):
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'graphllava' in model_name.lower():
        # Load GraphLLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None and pretrained_lora_path is None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading GraphLLaVA from base model...')
            model = GraphLLaVALlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, cache_dir=cache_dir,  **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional GraphLLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif 'lora' in model_name.lower() and model_base is not None and pretrained_lora_path is not None:
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading GraphLLaVA from base model...')
            model = GraphLLaVALlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, cache_dir=cache_dir,  **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, pretrained_lora_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')

            print('Loading additional GraphLLaVA weights...')
            projector_files = ["mm_projector_graph.bin", "mm_projector_text.bin", "mm_projector_image.bin", "mm_projector_node.bin", "mm_projector_nc.bin", "mm_projector_lp.bin", "mm_projector_shared_first_layer.bin", "mm_projector_shared.bin", "gate_network.bin", "common_projector.bin"]
            for file_name in projector_files:
                if not hasattr(model.get_model(), file_name.rsplit('.', 1)[0]):
                    continue
                full_local_path = os.path.join(model_path, file_name)
                if os.path.exists(full_local_path):
                    projector_weights = torch.load(full_local_path, map_location='cpu')
                    print(f"Load {file_name} from local path")
                else:
                    from huggingface_hub import hf_hub_download
                    model_path_hf = hf_hub_download(repo_id=model_path, filename=file_name)
                    projector_weights = torch.load(model_path_hf, map_location='cpu')
                    print(f"Load {file_name} from huggingface")

                # Cast to float16 and load into the model
                projector_weights = {k: v.to(torch.float16) for k, v in projector_weights.items()}
                model.load_state_dict(projector_weights, strict=False)



        elif model_base is not None:
            # this may be mm projector only
            print('Loading GraphLLaVA from base model...')
            # if 'mpt' in model_name.lower():
            #     if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
            #         shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
            #     tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            #     cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            #     model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            # else:
            #     tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            #     cfg_pretrained = AutoConfig.from_pretrained(model_path)
            #     model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            # if 'opt' in model_base:
            #     tokenizer = AutoTokenizer.from_pretrained(model_base)
            #     cfg_pretrained = AutoConfig.from_pretrained(model_path)
            #     model = LlagaOPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
            #                                                   cache_dir=cache_dir,
            #                                                   **kwargs)
            # else:
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            model = GraphLLaVALlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, cache_dir=cache_dir,
                                                              **kwargs)
            # model.get_model().initialize_graph_modules(cfg_pretrained)
            # if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
            #     mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            #     print("Load from local path")
            # else:
            #     from huggingface_hub import hf_hub_download
            #     model_path_hf = hf_hub_download(repo_id=model_path,  filename='mm_projector.bin')
            #     mm_projector_weights = torch.load(model_path_hf, map_location='cpu')
            #     print("Load from huggingface")
            # mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            # model.load_state_dict(mm_projector_weights, strict=False)
            #original CODE
            # projector_files = ["mm_projector_graph.bin", "mm_projector_text.bin", "mm_projector_image.bin", "mm_projector_node.bin", "mm_projector_nc.bin", "mm_projector_lp.bin", "mm_projector_shared_first_layer.bin", "mm_projector_shared.bin", "gate_network.bin", "common_projector.bin", "router.bin", "expert_projectors.bin"]
            # for file_name in projector_files:
            #     if not hasattr(model.get_model(), file_name.rsplit('.', 1)[0]):
            #         continue
            #     full_local_path = os.path.join(model_path, file_name)
            #     if os.path.exists(full_local_path):
            #         projector_weights = torch.load(full_local_path, map_location='cpu')
            #         print(f"Load {file_name} from local path")
            #     else:
            #         from huggingface_hub import hf_hub_download
            #         model_path_hf = hf_hub_download(repo_id=model_path, filename=file_name)
            #         projector_weights = torch.load(model_path_hf, map_location='cpu')
            #         print(f"Load {file_name} from huggingface")

            #     # Cast to float16 and load into the model
            #     projector_weights = {k: v.to(torch.float16) for k, v in projector_weights.items()}
            #     model.load_state_dict(projector_weights, strict=False)
            projector_files = ["mm_projector_graph.bin", "mm_projector_text.bin", "mm_projector_image.bin", 
                            "mm_projector_node.bin", "mm_projector_nc.bin", "mm_projector_lp.bin", 
                            "mm_projector_shared_first_layer.bin", "mm_projector_shared.bin", 
                            "gate_network.bin", "common_projector.bin", "router.bin", "expert_projectors.bin",
                            "task_token_embeddings.bin", "cross_task_attention.bin"]  # also load task_token_embeddings.bin

            for file_name in projector_files:
                # Special handling for task_token_embeddings
                if file_name == "task_token_embeddings.bin":
                    full_local_path = os.path.join(model_path, file_name)
                    if os.path.exists(full_local_path):
                        try:
                            task_tokens = torch.load(full_local_path, map_location='cpu')
                            print(f"Load {file_name} from local path")
                            
                            # Make sure the model has a task_token_embeddings attribute
                            if not hasattr(model.get_model(), "task_token_embeddings"):
                                # Create task_token_embeddings
                                hidden_dim = getattr(model.config, 'word_embed_proj_dim', 
                                                    getattr(model.config, 'hidden_size', 4096))
                                model.get_model().task_token_embeddings = nn.ParameterDict({
                                    'nc': nn.Parameter(torch.randn(1, hidden_dim, device=model.device, dtype=model.dtype)),
                                    'lp': nn.Parameter(torch.randn(1, hidden_dim, device=model.device, dtype=model.dtype))
                                })
                            
                            # Load the token embedding for each task
                            for task_name, token_emb in task_tokens.items():
                                if task_name in model.get_model().task_token_embeddings:
                                    model.get_model().task_token_embeddings[task_name].data.copy_(token_emb.to(model.dtype))
                            
                            print("Successfully loaded task token embeddings")
                        except Exception as e:
                            print(f"Error loading task token embeddings: {e}")
                        continue
                    else:
                        try:
                            from huggingface_hub import hf_hub_download
                            model_path_hf = hf_hub_download(repo_id=model_path, filename=file_name)
                            task_tokens = torch.load(model_path_hf, map_location='cpu')
                            print(f"Load {file_name} from huggingface")
                            
                            # Make sure the model has a task_token_embeddings attribute
                            if not hasattr(model.get_model(), "task_token_embeddings"):
                                # Create task_token_embeddings
                                hidden_dim = getattr(model.config, 'word_embed_proj_dim', 
                                                    getattr(model.config, 'hidden_size', 4096))
                                model.get_model().task_token_embeddings = nn.ParameterDict({
                                    'nc': nn.Parameter(torch.randn(1, hidden_dim, device=model.device, dtype=model.dtype)),
                                    'lp': nn.Parameter(torch.randn(1, hidden_dim, device=model.device, dtype=model.dtype))
                                })
                            
                            # Load the token embedding for each task
                            for task_name, token_emb in task_tokens.items():
                                if task_name in model.get_model().task_token_embeddings:
                                    model.get_model().task_token_embeddings[task_name].data.copy_(token_emb.to(model.dtype))
                            
                            print("Successfully loaded task token embeddings")
                        except Exception as e:
                            print(f"Error loading task token embeddings from HuggingFace: {e}")
                        continue
                
                # Handling for regular projectors
                if not hasattr(model.get_model(), file_name.rsplit('.', 1)[0]):
                    continue
                full_local_path = os.path.join(model_path, file_name)
                if os.path.exists(full_local_path):
                    projector_weights = torch.load(full_local_path, map_location='cpu')
                    print(f"Load {file_name} from local path")
                else:
                    from huggingface_hub import hf_hub_download
                    model_path_hf = hf_hub_download(repo_id=model_path, filename=file_name)
                    projector_weights = torch.load(model_path_hf, map_location='cpu')
                    print(f"Load {file_name} from huggingface")

                # Cast to float16 and load into the model
                projector_weights = {k: v.to(torch.float16) for k, v in projector_weights.items()}
                model.load_state_dict(projector_weights, strict=False)           
        else:
            # if 'mpt' in model_name.lower():
            #     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            #     model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            # else:
            #     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            #     model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = GraphLLaVALlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", cache_dir=cache_dir)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, cache_dir=cache_dir, **kwargs)


    if 'graphllava' in model_name.lower():
        mm_use_graph_start_end = getattr(model.config, "mm_use_graph_start_end", False)
        if mm_use_graph_start_end:
            tokenizer.add_tokens([DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len