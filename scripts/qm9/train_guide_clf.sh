for seed in 42 43 44;
do
cuda_visible_devices=2
for property in "alpha" "cv" "gap" "mu" "homo" "lumo";
do
    cmd="CUDA_VISIBLE_DEVICES=$cuda_visible_devices python train_classifier.py \
        --seed $seed \
        --num_layers 6 \
        --epoch 1200 \
        --max_len 9 \
        --cls_embed_size 64 \
        --e_embed_size 128 \
        --h_embed_size 128 \
        --log_dir guide_classifier \
        --target_property $property \
        --dataset qm9 \
        --wandb False"
    echo $cmd
    eval $cmd &
    cuda_visible_devices=`expr $cuda_visible_devices + 1`
done
wait
done