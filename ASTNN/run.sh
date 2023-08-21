DATASET="gcj"
CROSS=0
SPLIT="random0"
CUDA=0

python train.py --dataset $DATASET --cross $CROSS --split $SPLIT -cuda $CUDA 