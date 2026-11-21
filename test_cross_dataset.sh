device=0
epoch1=15
times=17


obj_list=("cookie" "carrot" "dowel")
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
        ## train on the VisA dataset
            base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_mv9_mvtec_3d
            save_dir=./my_exp/${times}/exps_${base_dir}_336_4/${obj_list[cls_id]}/weights/
            
             CUDA_VISIBLE_DEVICES=${device} python test3.py --dataset eye_pc_3d_rgb  \
            --data_path /root/autodl-tmp/Eyecandies_processed \
            --save_path ./my_exp/${times}/exps_${base_dir}_336_4/${obj_list[cls_id]} \
            --checkpoint_path ${save_dir}stage1/epoch_${epoch1}.pth \
            --features_list 24 --image_size 336 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
            --train_class ${obj_list[cls_id]} \
            --fusion_mode max \
            --w_global_p 0.7 \
            --w_patch_c 0.7 
        wait
        done
    done
done

