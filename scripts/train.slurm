#!/bin/bash

## Job Name
#SBATCH --job-name=jbirky
#SBATCH --account=astro
#SBATCH --partition=astro

#SBATCH --nodes=1   
#SBATCH --ntasks-per-node=28

#SBATCH --time=2-00:00:00
#SBATCH --mem=150G

## Specify the working directory for this job
#SBATCH --chdir=/gscratch/rory/jbirky/projects/spotgp/scripts/

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jbirky@uw.edu
#SBATCH --export=all

module load anaconda3_5.3
module load icc_18-impi_2018
python train_coeff.py --trial=5 --nsample=1000 --file_dir="../mox_hyak" --save_outputs=True