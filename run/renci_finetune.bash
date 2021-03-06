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
#     --train 4_train_enc --valid 4_val_enc  \
#     --llayers 9 --xlayers 5 --rlayers 5 \
#     --loadLXMERT /scratch/mmk11/snap/pretrained/model \
#     --dataPath /scratch/mmk11/data/renci/k \
#     --batchSize 5 --optim bert --lr 1e-3 --n_ent --epochs 100 --abla k_RENCI_meta \
#     --tqdm --output $output ${@:3}
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/rosmi.py \
    --train 1_train_enc --valid 1_val_enc  \
    --llayers 1 --xlayers 1 --rlayers 1 \
    --dataPath /scratch/mmk11/data/renci/k \
    --batchSize 1 --optim bert --lr 1e-3 --n_ent --epochs 200 --abla mapert_meta \
    --tqdm --output $output ${@:3}
