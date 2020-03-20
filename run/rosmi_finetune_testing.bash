# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/rosmi/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/rosmi.py \
    --train train --valid valid \
    --llayers 1 --xlayers 1 --rlayers 1 \
    --batchSize 50 --optim bert --lr 1e-3 --epochs 3000 \
    --output $output ${@:3}
