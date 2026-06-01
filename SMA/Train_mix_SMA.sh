!/bin/bash
for dataset in mix ; do
    python TrainQformer_with_Trainer_dino.py --vision_encoder_type clip --text_encoder_type clip --dataset_name $dataset --output_dir ./output/${dataset}_SMA --num_layers 6 --cross_att_frequency 2
    # wait
    # python -u Generate_emb.py --dataset $dataset --text_encoder_type clip
done
echo "Finish Training"
for subdataset in Reddit; do
    python -u load_model_dino.py --dataset $subdataset --vision_encoder_type clip --text_encoder_type clip --model_path ./output/${dataset}_SMA --output_name clip_mix
done

# done
# wait

# for dataset in "VideoGames" "RedditS";do
#     python TrainQformer_with_Trainer_dino.py \
#     --vision_encoder_type dino \
#     --text_encoder_type sbert \
#     --dataset_name $dataset \
#     --output_dir ./output/${dataset}_dino_sbert

#     python load_model_dino.py --dataset $dataset --vision_encoder_type dino --text_encoder_type sbert --model_path ./output/${dataset}_dino_sbert --output_name query_token_dino_sbert
# done