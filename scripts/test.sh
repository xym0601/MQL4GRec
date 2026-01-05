export NCCL_P2P_DISABLE=1
export WANDB_MODE=disabled 
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,4

Index_file=.index_lemb.json
Image_index_file=.index_vitemb.json

Tasks='seqrec,seqimage,item2image,image2item,fusionseqrec'
Valid_task1=seqrec
Valid_task2=seqimage

Datasets='Instruments'

load_model_name="/home/xym/data/mql4grec/logs/pretrain/checkpoint-21750"

OUTPUT_DIR=./logs/qk/$Datasets
mkdir -p $OUTPUT_DIR
log_file=$OUTPUT_DIR/train.log

torchrun --nproc_per_node=2 --master_port=2309 /home/xym/MQL4GRec1/finetune.py \
    --data_path /home/xym/MQL4GRec1/data \
    --dataset $Datasets \
    --output_dir $OUTPUT_DIR \
    --load_model_name $load_model_name \
    --per_device_batch_size 256 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --logging_step 10 \
    --max_his_len 20 \
    --prompt_num 4 \
    --patient 10 \
    --index_file $Index_file \
    --image_index_file $Image_index_file \
    --tasks $Tasks \
    --valid_task $Valid_task1 > $log_file \

results_file=$OUTPUT_DIR/results_${Valid_task1}_20.json
save_file=$OUTPUT_DIR/save_${Valid_task1}_20.json

torchrun --nproc_per_node=2 --master_port=2309 /home/xym/MQL4GRec1/test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path /home/xym/MQL4GRec1/data \
    --dataset $Datasets \
    --test_batch_size 64 \
    --num_beams 20 \
    --index_file $Index_file \
    --image_index_file $Image_index_file \
    --test_task $Valid_task1 \
    --results_file $results_file \
    --save_file $save_file \
    --filter_items > $log_file

results_file=$OUTPUT_DIR/results_${Valid_task2}_20.json
save_file=$OUTPUT_DIR/save_${Valid_task2}_20.json

torchrun --nproc_per_node=2 --master_port=2309 /home/xym/MQL4GRec1/test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path /home/xym/MQL4GRec1/data \
    --dataset $Datasets \
    --test_batch_size 64 \
    --num_beams 20 \
    --index_file $Index_file \
    --image_index_file $Image_index_file \
    --test_task $Valid_task2 \
    --results_file $results_file \
    --save_file $save_file \
    --filter_items > $log_file

python /home/xym/MQL4GRec1/ensemble.py \
    --output_dir $OUTPUT_DIR\
    --dataset $Datasets\
    --data_path /home/xym/MQL4GRec1/data\
    --index_file $Index_file\
    --image_index_file $Image_index_file\
    --num_beams 20 \
    --exp_name "ensemble"

