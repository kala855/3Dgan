#!/bin/bash


# To plot just one of the results. In this case the FP32 results
python ../eval_params_several_weights.py --weights_train \
     /gpfs/scratch/bsc28/bsc28459/3dgan_data/weights_fp32_200k_test/generator_params_generator_epoch_059.hdf5 \
     --real_data /gpfs/scratch/bsc28/bsc28459/3dgan_data/EleEscan_2_9.h5 \
     --out_dir ./output_test_eval_params_last/ \
     --xscales 100 \
     --labels fp32

# To print FP32, MP and AUTO results
#python ../eval_params_several_weights.py --weights_train \
#    /gpfs/scratch/bsc28/bsc28459/3dgan_data/weights_fp32_test_xscale_fix/generator_params_generator_epoch_029.hdf5 \
#    /gpfs/scratch/bsc28/bsc28459/3dgan_data/weights_mp_test_xscale_fix/generator_params_generator_epoch_029.hdf5 \
#    /gpfs/scratch/bsc28/bsc28459/3dgan_data/weights_auto_test_xscale_fixed/generator_params_generator_epoch_029.hdf5 \
#    --real_data /gpfs/scratch/bsc28/bsc28459/3dgan_data/EleEscan_1_9.h5 \
#    --out_dir ./output_test_eval_params_last/ \
#    --xscales 100 100 100 \
#    --labels fp32 mp auto
