#!/bin/bash
#SBATCH --account=m4334
#SBATCH --job-name=download
#SBATCH --email=savannah.ferretti@uci.edu
#SBATCH --mail-type=ALL
#SBATCH --constraint=cpu
#SBATCH --nodes=2
#SBATCH --time=06:00:00
#SBATCH --ntasks-per-node=4

# Load necessary modules
module load python

# Set path to your conda environemnt and activate it
CONDADIR='/global/homes/s/sferrett/.conda/envs/monsoon-sr'
source activate $CONDADIR

# Set environment variables needed for the Pyhton script
export AUTHOR='Savannah L. Ferretti'
export EMAIL='savannah.ferretti@uci.edu'
export SAVEDIR='/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/raw'
export YEARS='2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020'
export MONTHS='6 7 8'
export LATRANGE='5. 25.'
export LONRANGE='60. 90.'
export LEVRANGE='500. 1000.' 

# Change to the directory containing the script
SCRIPTDIR='/global/cfs/cdirs/m4334/sferrett/monsoon-sr/scripts'
cd $SCRIPTDIR

# Run the Python download script
srun python download.py

# Deactivate the conda environment
conda deactivate