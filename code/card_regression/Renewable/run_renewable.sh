export EXP_DIR=./results
export N_STEPS=1000
export SERVER_NAME=local
export RUN_NAME=test_run_lat
export LOSS=card_conditional
export TASK=$1
export N_SPLITS=1
export N_THREADS=4
export DEVICE_ID=1

export CAT_F_PHI=_cat_f_phi
export MODEL_VERSION_DIR=card_conditional_uci_results/${N_STEPS}steps/nn/${RUN_NAME}_${SERVER_NAME}/f_phi_prior${CAT_F_PHI}
# python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config Renewable/${TASK}.yml #--train_guidance_only
python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config $EXP_DIR/${MODEL_VERSION_DIR}/logs/ --test