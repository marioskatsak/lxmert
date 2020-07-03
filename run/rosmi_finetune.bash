# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=/scratch/mmk11/snap/rosmi/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
#--loadLXMERT snap/pretrained/model \
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/rosmi.py \
    --train train --valid valid --load /scratch/mmk11/snap/rosmi/BEST_1_t_NAME \
    --llayers 1 --xlayers 1 --rlayers 1 \
    --dataPath /scratch/mmk11/data/rosmi/ \
    --batchSize 1 --optim bert --lr 1e-3 --n_ent --epochs 100 --abla 1_t_NAME \
    --tqdm --output $output ${@:3}
