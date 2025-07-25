#!/bin/bash
#
#PBS -N Aquacrop_richards_daily
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=27
#SBATCH --mem=96gb
#SBATCH --time=72:00:00
#SBATCH --output=Aquacrop_richards_daily.out

cd $SLURM_SUBMIT_DIR
cd /home/kpanthi/dev/thesis/aquacrop-richards/aquacrop/
module load anaconda3/2023.09-0


# Run V2.py with the parameters
srun --cpu-bind=cores /home/kpanthi/.conda/envs/drl/bin/python compute_water_balance.py --use_richards True --hourly False --use_irrigation True