#!/bin/bash
#SBATCH --job-name="fp200k"
#SBATCH --workdir=.
#SBATCH --output=fp200k_epoch_60.out
#SBATCH --error=fp200k_epoch_60.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --qos=bsc_cs
#SBATCH --time=02-00:00:00
#SBATCH --exclusive

export OMP_NUM_THREADS=48

srun python ../EcalEnergyTrainNoHvdBSC.py \
    --datapath=/gpfs/scratch/bsc28/bsc28459/3dgan_data/EleEscan_*.h5 \
    --weightsdir=/gpfs/scratch/bsc28/bsc28459/3dgan_data/weights_fp32_200k_test \
    --batchsize 128 \
    --optimizer=Adam \
    --latentsize 200 \
    --analysis=False \
    --intraop 48 --interop 1 \
    --warmup 0 --nbepochs 60 \
    --last_epoch=45 \
    --generator_name=complete_generator_model_fp_epoch_60 \
    --discriminator_name=complete_discriminator_model_fp_epoch_60 \
    --train_history=/gpfs/scratch/bsc28/bsc28459/3dgan_data/weights_fp32_200k_test/train_history_60.pkl \
    --load_previous_model \
    --generator_model=/gpfs/scratch/bsc28/bsc28459/3dgan_data/weights_fp32_200k_test/complete_generator_model_fp_epoch_45.h5 \
    --discriminator_model=/gpfs/scratch/bsc28/bsc28459/3dgan_data/weights_fp32_200k_test/complete_discriminator_model_fp_epoch_45.h5
