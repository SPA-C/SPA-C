#!/bin/bash
#SBATCH -t 01-00:00:00
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -o AT.DToL.down.log
#SBATCH -J AT.DToL.down
#SBATCH --mail-type=BEGIN,END,FAIL

GENOME_ID="AT.DToL"
SPA_dir="SPA-C" # Directory of the Git repo
BASE_DIR="datasets" # Directory containing all datasets
MAIN_DIR="${BASE_DIR}/${GENOME_ID}"
HIC_DIR="${MAIN_DIR}/fastq"

# Downloading HiC
mkdir -p $HIC_DIR
wget "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR870/009/ERR8702799/ERR8702799_1.fastq.gz" -O ${HIC_DIR}/${GENOME_ID}_R1.fastq.gz
wget "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR870/009/ERR8702799/ERR8702799_2.fastq.gz" -O ${HIC_DIR}/${GENOME_ID}_R2.fastq.gz

# Downloading ASM
mkdir -p $MAIN_DIR/references
wget "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/933/208/065/GCA_933208065.1_ddAraThal4.1/GCA_933208065.1_ddAraThal4.1_genomic.fna.gz" -O ${MAIN_DIR}/references/${GENOME_ID}.ref.fna.gz

# Cutting contigs (for benchmarking purposes)
cd $MAIN_DIR/references
gzip -d ${GENOME_ID}.ref.fna.gz
awk -f ${BASE_DIR}/scaffold2ctg.awk ${GENOME_ID}.ref.fna
mv contigs.fasta ${GENOME_ID}.fa

# Preparing Juicer Template and launching it
cd $MAIN_DIR
cat ${SPA_dir}/paper/Preprocessings/JUICER.sh | sed "s/ctg_id/${GENOME_ID}/g" > JUICER.sh
sbatch JUICER.sh