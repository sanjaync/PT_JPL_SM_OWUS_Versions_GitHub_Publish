#!/bin/bash
#SBATCH --job-name=ptjpl_v3.2.1.1_full
#SBATCH --output=/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/output_L6_OWUS_MultiPET_v3.2.1.1/logs/full_run_%A_%a.out
#SBATCH --error=/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/output_L6_OWUS_MultiPET_v3.2.1.1/logs/full_run_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-45

# Load environment
module load miniforge3
conda activate ptjpl

# Define paths
BASE_DIR="/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm"
OUT_DIR="$BASE_DIR/output_L6_OWUS_MultiPET_v3.2.1.1"

# Create log directory
mkdir -p "$OUT_DIR/logs"

# Definitive site list
SITES=(
    "adelaideriver" "alicespringsmulga1" "alicespringsmulga2" "alpinepeatland" "boyagin" 
    "calperumchowilla" "capetribulation" "collie" "cowbay" "cumberlandplain" 
    "dalypasture" "dalyuncleared" "digbyplantation" "dryriver" "emerald" 
    "fletcherview" "foggdam" "gatumpasture" "gingin" "greatwesternwoodlands" 
    "howardsprings" "litchfield" "longreachmitchellgrassrangeland" "loxton" "myallvalea" 
    "myallvaleb" "otway" "reddirtmelonfarm" "ridgefield" "riggscreek" 
    "robsoncreek" "samfordecologicalresearchfacility" "silverplains" "snowgum" "sturtplains" 
    "titreeeast" "tumbarumba" "wallabycreek" "warra" "wellington" 
    "whroo" "wombatstateforest1" "wombatstateforest2" "yanco" "yarramundicontrol" 
    "yarramundiirrigated"
)

# Get current site name
SITE_NAME=${SITES[$SLURM_ARRAY_TASK_ID]}

echo "=========================================================="
echo "STARTING V3.2.1.1 PROCESS FOR: $SITE_NAME"
echo "Job ID: $SLURM_JOB_ID | Task: $SLURM_ARRAY_TASK_ID"
echo "Date: $(date)"
echo "=========================================================="

# STEP 1: Run the Model Processing (SPAC Analytical Inversion)
# This generates the _results.csv and _metrics.txt files
python "$BASE_DIR/ptjpl_owus_multipet_v3.2.1.1_run_site.py" "$SITE_NAME"

# STEP 2: Wait 5 seconds to ensure filesystem I/O is synced
sleep 5

# STEP 3: Run Plotting & Metrics Validation
# Only runs if the processing step was successful
if [ -d "$OUT_DIR/$SITE_NAME" ]; then
    echo "Processing successful. Generating plots..."
    python "$BASE_DIR/universal_plot_metrics.py" --base_dir "$OUT_DIR" --version "v3.2.1.1" --site "$SITE_NAME"
else
    echo "[ERROR] Processing failed for $SITE_NAME. Skipping plots."
    exit 1
fi

echo "=========================================================="
echo "SITE COMPLETED AT $(date)"
echo "=========================================================="