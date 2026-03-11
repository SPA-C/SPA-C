#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH -o MND.log
#SBATCH -J MND
#SBATCH --mem=64G  
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

module load bioinfo/samtools/1.21

samtools view -@8 -O SAM -F 1024 aligned/merged_dedup.bam | awk -v mnd=1 -f scripts/common/sam_to_pre.awk > aligned/merged_nodups.txt

squeue -j ${SLURM_JOBID}