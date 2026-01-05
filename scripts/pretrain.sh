export NCCL_P2P_DISABLE=1
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=2,3,4,6

Base_model=/home/xym/MQL4GRec/config/ckpt
Per_device_batch_size=1024
Learning_rate=1e-3
Epoch=30

Index_file=.index_lemb_256_dis_all_right.json
Image_index_file=.index_vitemb_256_dis_all_right.json

Tasks=seqrec,seqimage
Valid_task=seqrec

Datasets='Pet,Cell,Automotive,Tools,Toys,Sports'

OUTPUT_DIR=/home/xym/data/mql4grec/log/$Datasets/ckpt_b${Per_device_batch_size}_lr${Learning_rate}_${Tasks}/pretrain
mkdir -p $OUTPUT_DIR
log_file=$OUTPUT_DIR/pretrain.log

torchrun --nproc_per_node=4 --master_port=2309 pretrain.py \
    --data_path /home/xym/data/mql4grec \
    --pretrain_datasets $Datasets \
    --output_dir $OUTPUT_DIR \
    --base_model $Base_model \
    --per_device_batch_size $Per_device_batch_size \
    --learning_rate $Learning_rate \
    --epochs $Epoch \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --logging_step 10 \
    --train_data_mode 0 \
    --max_his_len 20 \
    --index_file $Index_file \
    --image_index_file $Image_index_file \
    --tasks $Tasks \
    --valid_task $Valid_task > $log_file

