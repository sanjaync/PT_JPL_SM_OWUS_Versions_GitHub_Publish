#!/bin/bash
#SBATCH --job-name=ptjpl_v13.3.1
#SBATCH --account=et97
#SBATCH --array=0-45
#SBATCH --partition=comp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --mail-user=sanjaync@monash.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs/ptjpl_v13.3.1_%A_%a.out
#SBATCH --error=logs/ptjpl_v13.3.1_%A_%a.err

module load miniforge3
conda activate ptjpl

export HDF5_USE_FILE_LOCKING=FALSE

WORKDIR=/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm
cd "$WORKDIR"
mkdir -p logs

SITE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" sites.txt)
SITE=$(echo "$SITE" | xargs)

echo "=========================================================="
echo "[$(date)] JOB: $SLURM_JOB_ID | TASK: $SLURM_ARRAY_TASK_ID"
echo "[$(date)] SITE: $SITE"
echo "[$(hostname)] v13.3.1 (SM-Only Grid Search | VOD + Single PET + OWUS Agent3 Scientific)"
echo "=========================================================="

python -u ptjpl_owus_single_pet_v13.3.1_run_site.py "$SITE"

echo "[$(date)] Finished."
