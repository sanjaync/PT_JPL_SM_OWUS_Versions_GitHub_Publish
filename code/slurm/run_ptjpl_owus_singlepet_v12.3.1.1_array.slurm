#!/bin/bash
#SBATCH --job-name=singlepet_v12.3.1.1
#SBATCH --output=/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/output_L6_OWUS_SinglePET_v12.3.1.1_DynamicBeta/logs/run_%A_%a.out
#SBATCH --error=/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/output_L6_OWUS_SinglePET_v12.3.1.1_DynamicBeta/logs/run_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-45

# Load environment
module load miniforge3
conda activate ptjpl

# Define paths
BASE_DIR="/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm"
OUT_DIR="$BASE_DIR/output_L6_OWUS_SinglePET_v12.3.1.1_DynamicBeta"

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

SITE_NAME=${SITES[$SLURM_ARRAY_TASK_ID]}

echo "STARTING v12.3.1.1 Analytical FOR: $SITE_NAME"
python "$BASE_DIR/ptjpl_owus_single_pet_v12.3.1.1_run_site.py" "$SITE_NAME"
# Optional: Run plotting if success
# if [ -d "$OUT_DIR/$SITE_NAME" ]; then
#     python "$BASE_DIR/universal_plot_metrics.py" --base_dir "$OUT_DIR" --version "v12.3.1.1" --site "$SITE_NAME"
# fi
