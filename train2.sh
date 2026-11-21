#!/bin/bash
device=0
epoch1=15
epoch2=10
times=17

obj_list=("carrot" "dowel" "cookie")
cls_ids=(0 1 2)
for cls_id in "${!cls_ids[@]}";do
    LOG=${save_dir}"res.log"
    echo ${LOG}
    echo ${cls_id} 
    depth=(9)
    n_ctx=(12)
    t_n_ctx=(4)
    for i in "${!depth[@]}";do
        for j in "${!n_ctx[@]}";do
            base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_mv9_mvtec_3d
            save_dir=./my_exp/${times}/exps_${base_dir}_336_4/${obj_list[cls_id]}/weights/

            CUDA_VISIBLE_DEVICES=${device} python train.py --dataset mvtec_pc_3d_rgb \
            --train_data_path /root/autodl-tmp/mvtec_3d_anomaly_detection \
            --save_path ${save_dir}stage1 \
            --features_list 24 --image_size 336  --print_freq 1 \
            --batch_size 4 \
            --epoch ${epoch1} --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
            --learning_rate 0.002 \
            --train_dataset_name ${obj_list[cls_id]} \
            --lvl_tilt 0.3 \
            --gamma_geo 0.2
            

            CUDA_VISIBLE_DEVICES=${device} python test3.py --dataset mvtec_pc_3d_rgb \
            --data_path /root/autodl-tmp/mvtec_3d_anomaly_detection \
            --save_path ./my_exp/${times}/exps_${base_dir}_336_4/${obj_list[cls_id]} \
            --stage1_checkpoint_path ${save_dir}stage1/epoch_${epoch1}.pth \
            --stage2_checkpoint_path ${save_dir}stage2/epoch_${epoch2}.pth \
            --features_list 24 --image_size 336 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
            --train_class ${obj_list[cls_id]} \
            --fusion_mode max \
            --w_global_p 0.7 \
            --w_patch_c 0.7 
        wait
        done
    done
done



obj_list=('PeppermintCandy' 'LicoriceSandwich' 'Confetto')
cls_ids=(0 1 2)
for cls_id in "${!cls_ids[@]}";do
    LOG=${save_dir}"res.log"
    echo ${LOG}
    echo ${cls_id} 
    depth=(9)
    n_ctx=(12)
    t_n_ctx=(4)
    for i in "${!depth[@]}";do
        for j in "${!n_ctx[@]}";do
            base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_mv9_eye_3d
            save_dir=./my_exp/${times}/exps_${base_dir}_336_4/${obj_list[cls_id]}/weightss/

            CUDA_VISIBLE_DEVICES=${device} python train.py --dataset eye_pc_3d_rgb \
            --train_data_path /root/autodl-tmp/Eyecandies_processed \
            --save_path ${save_dir}stage1 \
            --features_list 24 --image_size 336  --print_freq 1 \
            --batch_size 4 \
            --epoch ${epoch1} --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
            --learning_rate 0.002 \
            --train_dataset_name ${obj_list[cls_id]} \
            --lvl_tilt 0.3 \
            --gamma_geo 0.2
            
           
            CUDA_VISIBLE_DEVICES=${device} python test3.py --dataset eye_pc_3d_rgb \
            --data_path /root/autodl-tmp/Eyecandies_processed \
            --save_path ./my_exp/${times}/exps_${base_dir}_336_4/${obj_list[cls_id]} \
            --stage1_checkpoint_path ${save_dir}stage1/epoch_${epoch1}.pth \
            --stage2_checkpoint_path ${save_dir}stage2/epoch_${epoch2}.pth \
            --features_list 24 --image_size 336 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
            --train_class ${obj_list[cls_id]} \
            --fusion_mode max \
            --w_global_p 0.7 \
            --w_patch_c 0.7 
        done
    done
done