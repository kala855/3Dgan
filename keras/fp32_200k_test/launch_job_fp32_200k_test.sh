#!/bin/bash
#SBATCH --job-name="fp200k"
#SBATCH --workdir=.
#SBATCH --output=fp200k.out
#SBATCH --error=fp200k.err
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
    --warmup 0 --nbepochs 15 \
    --last_epoch=0 \
    --generator_name=complete_generator_model_fp_epoch_15 \
    --discriminator_name=complete_discriminator_model_fp_epoch_15 \
    --train_history=/gpfs/scratch/bsc28/bsc28459/3dgan_data/weights_fp32_200k_test/train_history.pkl

#    --load_previous_model \
#    --generator_model=/gpfs/scratch/bsc28/bsc28459/3dgan_data/weights_mp_to_restart/complete_generator_model_mp.h5 \
#    --discriminator_model=/gpfs/scratch/bsc28/bsc28459/3dgan_data/weights_mp_to_restart/complete_discriminator_model_mp.h5
