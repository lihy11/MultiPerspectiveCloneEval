mkdir saved_models

SPLIT="fun"
ABS=0
DATASET="gcj"

python run.py \
    --output_dir=saved_models \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=../datasetpre_train/${DATASET}/train_${SPLIT}.txt \
    --eval_data_file=../datasetpre_train/${DATASET}/valid_${SPLIT}.txt \
    --test_data_file=../datasetpre_train/${DATASET}/test_${SPLIT}.txt \
    --split ${SPLIT} \
    --abs ${ABS} \
    --dataset ${DATASET} \
    --epoch 1 \
    --code_length 1000 \
    --data_flow_length 128 \
    --train_batch_size 1 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 2>&1| tee saved_models/train_${SPLIT}_0.log
