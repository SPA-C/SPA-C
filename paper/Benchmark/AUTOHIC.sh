#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH -o <ID>.AutoHiC.log
#SBATCH -J <ID>.AutoHiC
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

# Specific to the Genotoul-bioinfo cluster
module load bioinfo/LASTZ/1.04.22 devel/python/Python-3.6.3  
module load bioinfo/3D-DNA/529ccf4
module load containers/Apptainer/1.2.5

Autohic_dir="AutoHiC" # Directory of AutoHic
apps="apptainer" # Path to apptainer images folder
ID="<ID>"

mkdir -p AutoHic

echo "Executing AutoHic"
apptainer exec $apps/AutoHic.sif /home/autohic/miniconda3/bin/conda run -n autohic \
	python3.9 ${Autohic_dir}/onehic.py \
	-hic "3DDNA/${ID}.final.hic" \
	-asy "3DDNA/${ID}.FINAL.assembly" \
	-autohic ${Autohic_dir} \
	-p "${Autohic_dir}/error_model.pth" \
	-out AutoHic

echo "Getting final Fasta"
run-asm-pipeline-post-review.sh \
	-r AutoHic/adjusted.assembly \
	references/${ID}.fa \
	aligned/merged_nodups.txt

rm -rf AutoHic/png

echo "Done."
# Printing execution time in log.
squeue -j ${SLURM_JOBID}
