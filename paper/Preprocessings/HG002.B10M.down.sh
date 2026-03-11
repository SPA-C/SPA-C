#!/bin/bash
#SBATCH -t 01-00:00:00
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -o HG002.B10M.down.log
#SBATCH -J HG002.B10M.down
#SBATCH --mail-type=BEGIN,END,FAIL

# Specific to the Genotoul-bioinfo cluster
module load tools/aws/2.13.14

GENOME_ID="HG002.B10M"
apps="apptainer" # Path to apptainer images folder
BASE_DIR="datasets" # Directory containing all datasets
SPA_dir="SPA-C" # Directory of the Git repo
MAIN_DIR="${BASE_DIR}/${GENOME_ID}"
HIC_DIR="${MAIN_DIR}/fastq"

# Reusing HG002 HiC dir
ln -s  ${BASE_DIR}/${GENOME_ID}/fastq $HIC_DIR

# Splitting the HG002.H1 assembly into 10Mb contigs
mkdir ${MAIN_DIR}/references
$apps/HiC.sif python -u ${SPA_dir}/paper/Preprocessings/ASM_breaker.py \
	--fasta ${BASE_DIR}/${GENOME_ID}/references/HG002.fa \
	--out ${MAIN_DIR}/references/${GENOME_ID}.fa \
	--size 10000000 \
	--simple-names

# Preparing Juicer Template and launching it
cd $MAIN_DIR
cat ${BASE_DIR}/JUICER.sh | sed "s/ctg_id/${GENOME_ID}/g" > JUICER.sh
sbatch JUICER.sh