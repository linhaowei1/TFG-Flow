cuda_visible_devices=5

taus=(0.0005 0.001 0.002 0.004)
rhos=(0.025 0.05 0.1 0.2)
for temperature in ${taus[@]};
do
for rho in ${rhos[@]};
do
for seed in 42 43 44;
do
    cmd="CUDA_VISIBLE_DEVICES=$cuda_visible_devices python sample_mols.py \
        --seed $seed \
        --num_layers 9 \
        --epoch 1200 \
        --max_len 9 \
        --cls_embed_size 64 \
        --e_embed_size 256 \
        --h_embed_size 256 \
        --sample_bs 256 \
        --sample_tot_num 4096 \
        --flow_ckpt ../flow_models/best.pth \
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
    eval $cmd 
    cuda_visible_devices=`expr $cuda_visible_devices + 1`
done
done
done