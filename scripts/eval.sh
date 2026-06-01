#!/bin/bash

emb=${1:-"clip"}
template=${2:-'ND'}
task_type=${3:-'nc'}
epoch=${4:-1}
dataset=${5:-'Movies'}

model_path=./checkpoints/Movies-Arts-RedditS-CD/mlaga-vicuna-7b-clip_mix-2-10-2-layer-mlp-TIQ_demo-projector_nc-lp_3
model_base="lmsys/vicuna-7b-v1.5-16k" #meta-llama/Llama-2-7b-hf
# model_base="meta-llama/Llama-2-7b-hf"
# mode="graphllava_llama_2" # use 'llaga_llama_2' for llama and "v1" for others
mode="v1"
dataset=${dataset} #test dataset
task=${task_type} #test task
emb=${emb}
use_hop=2
sample_size=10
template=${template} # or ND
output_path=./res/${dataset}_${template}_${emb}_${task}_${epoch}_data_task_generalization_yf.jsonl

python LLM_fix/eval/eval_pretrain.py \
--model_path ${model_path} \
--model_base ${model_base} \
--conv_mode  ${mode} \
--dataset ${dataset} \
--pretrained_embedding_type ${emb} \
--use_hop ${use_hop} \
--sample_neighbor_size ${sample_size} \
--answers_file ${output_path} \
--task ${task} \
--cache_dir ./checkpoint \
--template ${template}
wait
python ./LLM_fix/eval/eval_res.py --res_path ${output_path} --task ${task_type} --dataset ${dataset} --embedding_type ${emb}
