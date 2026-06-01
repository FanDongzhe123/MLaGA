#!/bin/bash

max_len=4096
sample_size=10

model=${1:-"vicuna"}
task=${2:-"nc"}
dataset=${3-"Movies"}
bs=${4:-16}
emb=${5:-"clip"}
template=${6:-"ND"}
epoch=${7:-1}
is_data_task_general=${8:-"False"}
is_cross_task_attention=${9:-"False"}
cross_attn_layers=${10:-1}
dataset_task_mapping=${11:-""}



if [ ${model} = "vicuna" ]; then
  use_hop=2
  template= ${template}
  projector_type="linear"
  prefix=mlaga-vicuna-7b-${emb}-${use_hop}-${sample_size}-${projector_type}-${template}-dual-projector-cross-attention-layer-${cross_attn_layers}
  model_base=lmsys/vicuna-7b-v1.5-16k
  mode="v1"
elif [ ${model} = "vicuna_2layer" ]; then
  use_hop=2
  template=${template}
  projector_type="2-layer-mlp"
  prefix=mlaga-vicuna-7b-${emb}-${use_hop}-${sample_size}-${projector_type}-${template}-dual-projector-cross-attention-layer-${cross_attn_layers}
  model_base=lmsys/vicuna-7b-v1.5-16k
  mode="v1"
elif [ ${model} = "vicuna_13b_2layer" ]; then
  use_hop=2
  template=${template}
  projector_type="2-layer-mlp"
  prefix=mlaga-vicuna-13b-${emb}-${use_hop}-${sample_size}-${projector_type}-${template}-dual-projector-cross-attention-layer-${cross_attn_layers}
  model_base=lmsys/vicuna-13b-v1.5-16k
  mode="v1"
elif [ ${model} = "llama" ]; then
  use_hop=2
  template=${template}
  projector_type="2-layer-mlp"
  prefix=mlaga-llama-2-7b-hf-${emb}-${use_hop}-${sample_size}-${projector_type}-dual-projector-cross-attention-layer-${cross_attn_layers}
  model_base=meta-llama/Llama-2-7b-hf
  mode="graphllava_llama_2"
fi


echo "PREFIX:  ${prefix}"

wandb online
echo python  MMGIT/train/train_mem.py \
--model_name_or_path ${model_base} \
--version ${mode} \
--cache_dir  ./checkpoint \
--pretrained_embedding_type ${emb} \
--tune_mm_mlp_adapter True \
--mm_use_graph_start_end False \
--mm_use_graph_patch_token False \
--bf16 True \
--output_dir  ./checkpoints/${dataset}/${prefix}_${task}_${epoch} \
--num_train_epochs ${epoch} \
--per_device_train_batch_size ${bs} \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length ${max_len} \
--gradient_checkpointing True \
--lazy_preprocess True \
--report_to wandb \
--use_hop ${use_hop} \
--sample_neighbor_size ${sample_size} \
--mm_projector_type ${projector_type} \
--use_task ${task} \
--use_dataset ${dataset} \
--template ${template} \
--is_general_model False \
--is_data_task_general ${is_data_task_general} \
--is_cross_task_attention ${is_cross_task_attention} \
--dataset_task_mapping ${dataset_task_mapping} \
--cross_attn_layers ${cross_attn_layers} \


python  MMGIT/train/train_mem.py \
--model_name_or_path ${model_base} \
--version ${mode} \
--cache_dir  ./checkpoint \
--pretrained_embedding_type ${emb} \
--tune_mm_mlp_adapter True \
--mm_use_graph_start_end False \
--mm_use_graph_patch_token False \
--bf16 True \
--output_dir  ./checkpoints/${dataset}/${prefix}_${task}_${epoch} \
--num_train_epochs ${epoch} \
--per_device_train_batch_size ${bs} \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length ${max_len} \
--gradient_checkpointing True \
--lazy_preprocess True \
--report_to wandb \
--use_hop ${use_hop} \
--sample_neighbor_size ${sample_size} \
--mm_projector_type ${projector_type} \
--use_task ${task} \
--use_dataset ${dataset} \
--template ${template} \
--is_general_model False \
--is_data_task_general ${is_data_task_general} \
--is_cross_task_attention ${is_cross_task_attention} \
--dataset_task_mapping ${dataset_task_mapping} \
--cross_attn_layers ${cross_attn_layers} \
