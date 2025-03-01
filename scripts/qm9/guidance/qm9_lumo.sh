cuda_visible_devices=0

taus=(0.016 0.008 0.004 0.002)
rhos=(0.01 0.02 0.04 0.08)
for temperature in ${taus[@]};
do
cuda_visible_devices=0
for rho in ${rhos[@]};
do
for seed in 42;
do
    cmd="CUDA_VISIBLE_DEVICES=$cuda_visible_devices python sample_mols.py \
        --seed $seed \
        --num_layers 9 \
        --epoch 1200 \
        --max_len 9 \
        --cls_embed_size 64 \
        --e_embed_size 256 \
        --h_embed_size 256 \
        --sample_bs 128 \
        --sample_tot_num 4096 \
        --flow_ckpt ./storage/flow_model.pth \
        --target_property lumo \
        --dataset qm9 \
        --wandb False \
        --k 512 \
        --temperature $temperature \
        --n_iter 4 \
        --n_recur 1 \
        --rho $rho \
        --mu $rho \
        --gamma 0.0 \
        --num_eps 1 \
        --guidance_weight 1 \
        --wandb False"
    echo $cmd
    eval $cmd &
    cuda_visible_devices=$((cuda_visible_devices+1))
done
done
wait
done