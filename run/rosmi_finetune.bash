# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=/scratch/mmk11/snap/rosmi/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
#--loadLXMERT snap/pretrained/model \
# CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PYTHONPATH:./src \
#     python src/tasks/rosmi.py \
#     --train 0_train --valid 0_val \
#     --llayers 2 --xlayers 1 --rlayers 2 \
#     --dataPath /scratch/mmk11/data/rosmi/7_easy_train \
#     --batchSize 32 --optim bert --lr 1e-4 --n_ent --epochs 130 --abla random_t_NAME \
#     --tqdm --output $output ${@:3}
    # See Readme.md for option details.
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
        python src/tasks/rosmi.py \
        --valid val --load /scratch/mmk11/snap/rosmi/BEST_k_RENCI_NAME \
        --llayers 1 --xlayers 1 --rlayers 1 --cross \
        --dataPath /scratch/mmk11/data/rosmi/7_easy_train \
        --batchSize 1 --optim bert --lr 1e-3 --n_ent --epochs 80 --abla hybrid_rule_based \
        --tqdm --output $output ${@:3}
