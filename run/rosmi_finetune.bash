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
    --train train --valid valid \
    --llayers 0 --xlayers 0 --rlayers 0 \
    --fromScratch \
    --dataPath /scratch/mmk11/data/rosmi/ \
    --batchSize 20 --optim bert --lr 1e-3 --epochs 250 --abla t_scr_trans_Boxes \
    --tqdm --output $output ${@:3}
