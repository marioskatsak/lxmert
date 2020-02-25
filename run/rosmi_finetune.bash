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
    --train train --valid valid  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERT snap/pretrained/model \
    --dataPath /scratch/mmk11/data/rosmi/ \
    --batchSize 1 --optim bert --lr 5e-1 --epochs 100 \
    --tqdm --output $output ${@:3}
