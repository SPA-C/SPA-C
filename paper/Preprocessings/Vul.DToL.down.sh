#!/bin/bash
#SBATCH -t 01-00:00:00
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -o Vul.DToL.down.log
#SBATCH -J Vul.DToL.down
#SBATCH --mail-type=BEGIN,END,FAIL

GENOME_ID="Vul.DToL"
SPA_dir="SPA-C" # Directory of the Git repo
BASE_DIR="datasets" # Directory containing all datasets
MAIN_DIR="${BASE_DIR}/${GENOME_ID}"
HIC_DIR="${MAIN_DIR}/fastq"

# Downloading HiC
mkdir -p $HIC_DIR
wget "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR128/084/ERR12862084/ERR12862084_1.fastq.gz" -O ${HIC_DIR}/${GENOME_ID}_R1.fastq.gz
wget "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR128/084/ERR12862084/ERR12862084_2.fastq.gz" -O ${HIC_DIR}/${GENOME_ID}_R2.fastq.gz

# Downloading ASM
mkdir -p $MAIN_DIR/references
wget "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/964/106/825/GCA_964106825.2_mVulVul1.hap1.2/GCA_964106825.2_mVulVul1.hap1.2_genomic.fna.gz" -O ${MAIN_DIR}/references/${GENOME_ID}.ref.fna.gz

# Cutting contigs (for benchmarking purposes)
cd $MAIN_DIR/references
gzip -d ${GENOME_ID}.ref.fna.gz
awk -f ${BASE_DIR}/scaffold2ctg.awk ${GENOME_ID}.ref.fna
mv contigs.fasta ${GENOME_ID}.fa

# Preparing Juicer Template and launching it
cd $MAIN_DIR
cat ${SPA_dir}/paper/Preprocessings/JUICER.sh | sed "s/ctg_id/${GENOME_ID}/g" > JUICER.sh
sbatch JUICER.sh