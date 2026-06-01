for epoch in 4; do
    for dataset in "CD" "Art500K"; do
        ./scripts/eval_data_task_generalization_cross_task_attention_lr.sh clip_mix TIQ_demo lp $epoch $dataset
    done
done
for epoch in 4; do
    for dataset in "Movies" "VideoGames"; do
        ./scripts/eval_data_task_generalization_cross_task_attention_lr.sh clip_mix TIQ_demo nc $epoch $dataset
    done
done