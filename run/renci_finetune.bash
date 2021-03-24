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
    --train 4_train_enc --valid 4_val_enc \
    --llayers 2 --xlayers 1 --rlayers 2 \
    --dataPath /scratch/mmk11/data/renci/k \
    --batchSize 32 --optim bert --lr 1e-4 --n_ent --epochs 140 --abla k_RENCI_meta \
    --tqdm --output $output ${@:3}
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
#     python src/tasks/rosmi.py \
#     --valid valid --load /scratch/mmk11/snap/rosmi/BEST_2_t_NAME \
#     --llayers 1 --xlayers 1 --rlayers 1\
#     --dataPath /scratch/mmk11/data/rosmi \
#     --batchSize 20 --optim bert --lr 1e-3 --n_ent --epochs 80 --abla BEST_2_t_NAME \
#     --tqdm --output $output ${@:3}
