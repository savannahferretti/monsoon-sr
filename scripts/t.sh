#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100GB
#SBATCH --time=6:00:00
#SBATCH --constraint=cpu
#SBATCH --licenses=cfs
#SBATCH --qos=regular
#SBATCH --account=m4334
#SBATCH --job-name=t
#SBATCH --output=t.log
#SBATCH --mail-user=savannah.ferretti@uci.edu
#SBATCH --mail-type=ALL

# Load necessary modules
module load python conda

# Activate the Conda environment
CONDADIR='/global/homes/s/sferrett/.conda/envs/monsoon-sr'
source activate $CONDADIR

# Navigate to the directory where the Python script is
SCRIPTDIR='/global/cfs/cdirs/m4334/sferrett/monsoon-sr/scripts'
cd $SCRIPTDIR

# Execute the Python script
srun --cpu-bind=cores python t.py 2>&1

# Deactivate the Conda environment
conda deactivate