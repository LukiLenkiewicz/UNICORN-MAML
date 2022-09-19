#!/bin/bash

set -ex

DATASET=CUB
PRE_COMMON_PATH="pretrained/${DATASET}/con-pre.pth"
BKB_CLASS=Conv4
N_SHOT=1

BASE_CMD="python train_fsl.py --way 5 --eval_way 5 --lr_scheduler step  --lr_mul 10 --backbone_class ${BKB_CLASS} \
  --dataset ${DATASET} --gpu ${CUDA_VISIBLE_DEVICES} --query 15 --step_size 20 --gamma 0.1 \
  --lr 0.001 --shot ${N_SHOT} --eval_shot ${N_SHOT} --temperature 0.5 --gd_lr 0.05 \
  --eval_interval 1 --num_eval_episodes 100"

# pretrain MAML
yes | $BASE_CMD --model_class MAML --max_epoch 10 --para_init $PRE_COMMON_PATH --suffix pretrained_MAML

PRE_MAML_PATH="checkpoints/${DATASET}-MAML-${BKB_CLASS}-05w0${N_SHOT}s15q-Pre/20_0.1_lr0.001_sgd_InLR0.05InIter5_step_t0.5-Mul10.0_pretrained_MAML"


for PRE_PATH in ${PRE_COMMON_PATH} ${PRE_MAML_PATH}; do
  if [ "$PRE_PATH" = "$PRE_COMMON_PATH" ]; then
      PRE_SUFFIX="pre_common"
  else
      PRE_SUFFIX="pre_maml"
  fi

  for HN_LEN in 1 2 3; do
    for HN_WIDTH in 128 256 512; do
      for OPT in "sgd" "adam"; do
        for FREEZE_OPTION in "--hm_freeze_backbone" ""; do
          SUFFIX="${PRE_SUFFIX}_${HN_LEN}_${HN_WIDTH}_${OPT}_${FREEZE_OPTION}"
          $BASE_CMD --para_init $PRE_PATH --hm_hn_len $HN_LEN --hm_hn_width ${HN_WIDTH} --optimizer_class ${OPT} $FREEZE_OPTION --suffix $SUFFIX &
        done

        wait #for parallelism
      done
    done
  done
done









