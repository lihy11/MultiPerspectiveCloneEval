mkdir saved_models

model=microsoft/unixcoder-base
SPLIT="random"
ABS=0
DATASET="cbcb"
python run.py \
    --output_dir=./saved_models \
    --model_name_or_path=${model} \
    --do_train \
    --do_eval \
    --train_data_file=../dataset/pre_train/${DATASET}/train_${SPLIT}_test.txt \
    --eval_data_file=../dataset/pre_train/${DATASET}/valid_${SPLIT}.txt \
    --test_data_file=../dataset/pre_train/${DATASET}/test_${SPLIT}.txt \
    --split ${SPLIT} \
    --abs ${ABS} \
    --dataset ${DATASET} \
    --num_train_epochs 1 \
    --block_size 512 \
    --train_batch_size 3 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 2>&1| tee saved_models/train_${DATASET}_${SPLIT}_${ABS}.log


