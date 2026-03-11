#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH -o <ID>.3DDNA.log
#SBATCH -J <ID>.3DDNA
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

# Specific to the Genotoul-bioinfo cluster
module load bioinfo/LASTZ/1.04.22 devel/python/Python-3.6.3
module load bioinfo/3D-DNA/529ccf4

mkdir -p 3DDNA
cd 3DDNA

run-asm-pipeline.sh \
	-m haploid \
	../references/<ID>.fa \
	../aligned/merged_nodups.txt

# Printing execution time in log.
squeue -j ${SLURM_JOBID}