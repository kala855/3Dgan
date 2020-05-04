#!/bin/bash
#SBATCH --job-name="bash_pdf"
#SBATCH --workdir=.
#SBATCH --output=bash_pdf.out
#SBATCH --error=bash_pdf.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --qos=debug
#SBATCH --time=00-02:00:00
#SBATCH --exclusive

#export OMP_NUM_THREADS=48
./bash_eval_params.sh
