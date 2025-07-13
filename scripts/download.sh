#!/bin/bash
#SBATCH --account=m4334
#SBATCH --job-name=download
#SBATCH --output=%x.log
#SBATCH --mail-user=savannah.ferretti@uci.edu
#SBATCH --mail-type=ALL
#SBATCH --qos=regular
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --licenses=cfs

# Load necessary modules
module load python
module load conda

# Set path to your conda environemnt and activate it
CONDADIR='/global/homes/s/sferrett/.conda/envs/monsoon-sr'
source activate $CONDADIR

# Change to the directory containing the script
SCRIPTDIR='/global/cfs/cdirs/m4334/sferrett/monsoon-sr/scripts'
cd $SCRIPTDIR

# Run the Python download script
srun --cpu-bind=cores python download.py 2>&1

# Deactivate the conda environment
conda deactivate