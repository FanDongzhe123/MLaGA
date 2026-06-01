from typing import List, Optional, Tuple, Union

import pdb
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

import sys
sys.path.append("..")
sys.path.append("../..")
from ..GraphLLava_arch import GraphLLaVAMetaModel, GraphLLaVAMetaForCausalLM
from MMGIT.constants import IGNORE_INDEX


class GraphLLaVAConfig(LlamaConfig):
    model_type = "graphllava"


class GraphLLaVALlamaModel(GraphLLaVAMetaModel, LlamaModel):
    config_class = GraphLLaVAConfig

    def __init__(self, config: LlamaConfig):
        super(GraphLLaVALlamaModel, self).__init__(config)


class GraphLLaVALlamaForCausalLM(LlamaForCausalLM, GraphLLaVAMetaForCausalLM):
    config_class = GraphLLaVAConfig

    def __init__(self, config):
        # super(LlamaForCausalLM, self).__init__(config)
        LlamaForCausalLM.__init__(self, config)
        self.model = GraphLLaVALlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graph_emb: Optional[torch.FloatTensor] = None,
        image_emb: Optional[torch.FloatTensor] = None,
        text_emb: Optional[torch.FloatTensor] = None,
        node_emb: Optional[torch.FloatTensor] = None,
        task_type: Optional[list] = None,
        graph: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, graph_emb, image_emb, text_emb, node_emb, graph, task_type)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # cache_position=None 
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):  
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graph_emb": kwargs.get("graph_emb", None),
                "image_emb": kwargs.get("image_emb", None),
                "text_emb": kwargs.get("text_emb", None),
                "node_emb": kwargs.get("node_emb", None),
                "task_type": kwargs.get("task_type", None),
                "graph": kwargs.get("graph", None),
            }
        )
        return model_inputs

AutoConfig.register("graphllava", GraphLLaVAConfig)
AutoModelForCausalLM.register(GraphLLaVAConfig, GraphLLaVALlamaForCausalLM)
