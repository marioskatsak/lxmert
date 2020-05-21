# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=/scratch/mmk11/snap/rosmi/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
#--loadLXMERT snap/pretrained/model \
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/rosmi.py \
    --train train --valid valid \
    --llayers 0 --xlayers 0 --rlayers 0 \
    --dataPath /scratch/mmk11/data/rosmi/ \
    --batchSize 20 --optim bert --lr 1e-3 --n_ent --epochs 200 --abla trans_Feat_Boxes_Names \
    --tqdm --output $output ${@:3}
