# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=/scratch/mmk11/snap/rosmi/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
#--loadLXMERT snap/pretrained/model \
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/rosmi.py \
    --valid 0_val --load /scratch/mmk11/snap/rosmi/BEST_2_t_NAME \
    --llayers 1 --xlayers 1 --rlayers 1\
    --dataPath /scratch/mmk11/data/rosmi/7_easy_train \
    --batchSize 20 --optim bert --lr 1e-3 --n_ent --epochs 80 --abla BEST_2_t_NAME \
    --tqdm --output $output ${@:3}
