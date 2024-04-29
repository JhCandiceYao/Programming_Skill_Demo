#!/bin/bash
#SBATCH --job-name=sg
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=180GB
#SBATCH --time=24:00:00
#SBATCH --array=9-10
#SBATCH --output=/scratch/jy3440/MOTIFS/errors/rerun_%A_%a.out
#SBATCH --error=/scratch/jy3440/MOTIFS/errors/rerun_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jy3440@nyu.edu

module purge
module load r/gcc/4.1.2

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /scratch/jy3440/MOTIFS
Rscript semgram_csv.R