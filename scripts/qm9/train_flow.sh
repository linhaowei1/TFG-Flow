cuda_visible_devices=0
for seed in 42 43 44;
do
cmd="CUDA_VISIBLE_DEVICES=$cuda_visible_devices python train_flow.py \
    --seed $seed \
    --num_layers 9 \
    --epoch 1200 \
    --max_len 9 \
    --cls_embed_size 64 \
    --e_embed_size 256 \
    --h_embed_size 256 \
    --dataset qm9 \
    --wandb False"
echo $cmd
eval $cmd &
cuda_visible_devices=`expr $cuda_visible_devices + 1`
done
wait