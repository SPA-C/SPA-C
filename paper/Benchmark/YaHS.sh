#!/bin/bash  
#SBATCH --cpus-per-task=4  
#SBATCH -o <ID>.YaHS.log
#SBATCH -J <ID>.YaHS
#SBATCH --mem=64G  
#SBATCH --time=01:00:00  
#SBATCH --mail-type=BEGIN,END,FAIL

# Specific to the Genotoul-bioinfo cluster
module load bioinfo/YaHS/1.2.2  

mkdir yahs.out

yahs \
	-o yahs.out/yahs.out \
	references/<ID>.fa \
	aligned/merged_dedup.bam
	
# Printing execution time in log.
squeue -j ${SLURM_JOBID}