# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=/scratch/mmk11/snap/rosmi/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
#--loadLXMERT snap/pretrained/model \
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
#     python src/tasks/rosmi.py \
#     --train train --valid valid \
#     --llayers 1 --xlayers 1 --rlayers 1 \
#     --dataPath /scratch/mmk11/data/renci/k \
#     --batchSize 64 --optim bert --lr 1e-3 --n_ent --epochs 80 --abla load_k_RENCI_NAME \
#     --tqdm --output $output ${@:3}
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/rosmi.py \
    --train train --valid valid --load /scratch/mmk11/snap/rosmi/BEST_load_k_RENCI_NAME \
    --llayers 1 --xlayers 1 --rlayers 1 --single \
    --dataPath /scratch/mmk11/data/renci/k \
    --batchSize 1 --optim bert --lr 1e-3 --n_ent --epochs 80 --abla load_k_RENCI_NAME \
    --tqdm --output $output ${@:3}
