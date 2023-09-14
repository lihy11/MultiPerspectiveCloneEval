DATASET="cbcb"
CROSS=0
SPLIT="random0"
CUDA=0
mkdir ./cache/
python run.py --dataset $DATASET --cross $CROSS --split $SPLIT --cuda $CUDA 