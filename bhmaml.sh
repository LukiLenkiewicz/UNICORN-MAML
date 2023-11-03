DATASET=cross_char
PRE_COMMON_PATH="pretrained/${DATASET}/con-pre.pth"
BKB_CLASS=Conv4
MODEL_CLASS=BHyperMAML
N_SHOT=1


# cd UNICORN-MAML

python train_fsl.py --model_class ${MODEL_CLASS} --way 5 --eval_way 5 --lr_scheduler step  --lr_mul 10 --backbone_class ${BKB_CLASS} --dataset ${DATASET} --query 15 --step_size 20 --gamma 0.1 --lr 0.001 --shot ${N_SHOT} --eval_shot ${N_SHOT} --temperature 1 --gd_lr 0.01 --eval_interval 1 --num_eval_episodes 100 --bm_chunk_emb_size 8 --max_epoch 200 --hm_hn_len 2 --hm_hn_width 512 --inner_iters 5 --step_size 5 --bm_maml_first
