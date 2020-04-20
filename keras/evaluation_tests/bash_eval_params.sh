#!/bin/bash

# Here we define the base folder of the weights to be tested, in
# case you need to take into account just one specific weight
# file define it completely in the TO_CHECK variable
#WEIGHTS_BASE_FOLDER="/gpfs/scratch/bsc28/bsc28459/3dgan_data/weights_fp32_test_xscale_fix/"
#WEIGHTS_BASE_FOLDER="/gpfs/scratch/bsc28/bsc28459/3dgan_data/weights_mp_test_xscale_fix/"
WEIGHTS_BASE_FOLDER="/gpfs/scratch/bsc28/bsc28459/3dgan_data/weights_auto_test_xscale_fixed/"
#WEIGHTS_TRAIN="generator_params_generator_epoch_005.hdf5"
#FULL_WEIGHTS_PATH=$WEIGHTS_BASE_FOLDER$WEIGHTS_TRAIN

# Here we define the base folder to look for the real data to be
# compared with the results generated using the previous loaded
# weights
REAL_DATA_BASE="/gpfs/scratch/bsc28/bsc28459/3dgan_data/"
REAL_DATA_FILE="EleEscan_1_9.h5"
#REAL_DATA_FILE="EleEscan_1_1.h5"
FULL_REAL_DATA_FILE=$REAL_DATA_BASE$REAL_DATA_FILE

# Here we define the base folder to put the generated plots
RESULTS_BASE_FOLDER="/gpfs/scratch/bsc28/bsc28459/3dgan_data/results_auto_test_xscale_fix/"
mkdir -p $RESULTS_BASE_FOLDER

#RESULTS_FILE="EleEscan_1_1_no_horovod_epoch_005_test_params.pdf"
#FULL_RESULTS_FILE=$RESULTS_BASE_FOLDER$RESULTS_FILE
RESULTS_FILE_TEST="EleEscan_1_9_auto_test_xscale_fix_epoch_"
FILE_EXTENSION=".pdf"

TO_CHECK="generator_params_generator*"
i=0
for f in $WEIGHTS_BASE_FOLDER$TO_CHECK
do
    echo "processing $f file ..."
    python ../eval_params.py --weights_train $f --real_data $FULL_REAL_DATA_FILE \
        --out_dir $RESULTS_BASE_FOLDER$RESULTS_FILE_TEST$i/ \
        --xscales 100
    i=$((i+1))
done

#python ../eval_params.py --weights_train \
#    /gpfs/scratch/bsc28/bsc28459/3dgan_data/weights_fp32_test_xscale_fix/generator_params_generator_epoch_011.hdf5 \
#    --real_data /gpfs/scratch/bsc28/bsc28459/3dgan_data/EleEscan_1_9.h5 \
#    --out_dir ./output_xscales_fp32/ \
#    --xscales 100

#python eval_params.py --weights_train $FULL_WEIGHTS_PATH \
#    --real_data $FULL_REAL_DATA_FILE --out_file $FULL_RESULTS_FILE
