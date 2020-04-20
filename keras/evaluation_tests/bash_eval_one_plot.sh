#!/bin/bash

python ../eval_params.py --weights_train \
    /gpfs/scratch/bsc28/bsc28459/3dgan_data/weights_fp32_test_xscale_fix/generator_params_generator_epoch_011.hdf5 \
    --real_data /gpfs/scratch/bsc28/bsc28459/3dgan_data/EleEscan_1_9.h5 \
    --out_dir ./output_xscales_fp32/ \
    --xscales 100
#    --norm True
