# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=/scratch/mmk11/snap/rosmi/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
#--loadLXMERT snap/pretrained/model \
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/rosmi.py \
    --train train --valid valid  \
    --llayers 1 --xlayers 1 --rlayers 1 \
    --dataPath /scratch/mmk11/data/rosmi/ \
    --batchSize 32 --optim bert --lr 1e-3 --epochs 5000 \
    --tqdm --output $output ${@:3}
