#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH -o <ID>.pinhic.log
#SBATCH -J <ID>.pinhic
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir pinhic.out
ID="<ID>"
bin_dir="pin_hic/bin"

${bin_dir}/pin_hic_it \
	-O pinhic.out/ \
	-x references/${ID}.fa.fai \
	-r references/${ID}.fa \
	aligned/merged_dedup.bam

# Printing execution time in log.
squeue -j ${SLURM_JOBID}
