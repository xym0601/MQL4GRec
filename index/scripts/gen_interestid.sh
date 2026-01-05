export CUDA_VISIBLE_DEVICES=4

# 你的数据根目录
DATA_ROOT="/home/xym/data/mql4grec"

# 预训练通常涉及的所有数据集（根据你的实际情况修改）
# 常见的 MQL4GRec 数据集列表:
Datasets='Instruments Arts Games Pet Cell Automotive Tools Toys Sports'

for Dataset in $Datasets; do
    echo "=================================================="
    echo "Generating Interest IDs for: $Dataset"
    echo "=================================================="
    
    python -u /home/xym/MQL4GRec/index/gen_interestid.py \
      --dataset $Dataset \
      --data_root $DATA_ROOT
      
    if [ $? -ne 0 ]; then
        echo "Error processing $Dataset"
        exit 1
    fi
done

echo "All datasets processed successfully."