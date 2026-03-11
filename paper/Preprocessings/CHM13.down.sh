#!/bin/bash
#SBATCH -t 01-00:00:00
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -o CHM13.down.log
#SBATCH -J CHM13.down
#SBATCH --mail-type=BEGIN,END,FAIL

module load tools/aws/2.13.14

GENOME_ID="CHM13"
SPA_dir="SPA-C" # Directory of the Git repo
BASE_DIR="datasets" # Directory containing all datasets
MAIN_DIR="${BASE_DIR}/${GENOME_ID}"
HIC_DIR="${MAIN_DIR}/fastq"

# Downloading HiC
mkdir -p $HIC_DIR
aws s3 --no-sign-request cp s3://human-pangenomics/T2T/CHM13/arima/CHM13.rep1_lane1_R1.fastq.gz $HIC_DIR/
aws s3 --no-sign-request cp s3://human-pangenomics/T2T/CHM13/arima/CHM13.rep1_lane1_R2.fastq.gz $HIC_DIR/

# Downloading ASM
mkdir -p $MAIN_DIR/references
wget "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/analysis_set/chm13v2.0_noY.fa.gz" -O ${MAIN_DIR}/references/${GENOME_ID}.ref.fna.gz
cd $MAIN_DIR/references
gzip -d ${GENOME_ID}.ref.fna.gz
ln -s ${GENOME_ID}.ref.fna ${GENOME_ID}.fa

# Preparing Juicer Template and launching it
cd $MAIN_DIR
cat ${SPA_dir}/paper/Preprocessings/JUICER.sh | sed "s/ctg_id/${GENOME_ID}/g" > JUICER.sh
sbatch JUICER.sh