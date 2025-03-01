cuda_visible_devices=(3 4 5 6)
taus=(0.5 1 2 4)
rhos=(0.0025 0.005 0.01 0.02)
for temperature in ${taus[@]};
do
for rho in ${rhos[@]};
do
cuda_idx=0
for seed in 44;
do
    cuda_visible_device=${cuda_visible_devices[$cuda_idx]}
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
        --target_property mu \
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
    cuda_idx=`expr $cuda_idx + 1`
done
done
wait
done