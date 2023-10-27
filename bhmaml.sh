DATASET=CUB
PRE_COMMON_PATH="pretrained/${DATASET}/con-pre.pth"
BKB_CLASS=Conv4
N_SHOT=1

python train_fsl.py --model_class BHyperMAML --way 5 --eval_way 5 --lr_scheduler step  --lr_mul 10 --backbone_class ${BKB_CLASS} --dataset ${DATASET} --query 15 --step_size 20 --gamma 0.1 --lr 0.001 --shot ${N_SHOT} --eval_shot ${N_SHOT} --temperature 0.5 --gd_lr 0.05 --eval_interval 1 --num_eval_episodes 100 --bm_chunk_emb_size 8