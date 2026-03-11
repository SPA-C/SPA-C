#!/bin/bash
#SBATCH -t 01-00:00:00
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -o HG002.down.log
#SBATCH -J HG002.down
#SBATCH --mail-type=BEGIN,END,FAIL

module load tools/aws/2.13.14

GENOME_ID="HG002"
BASE_DIR="datasets" # Directory containing all datasets
MAIN_DIR="${BASE_DIR}/${GENOME_ID}"
HIC_DIR="${MAIN_DIR}/fastq"

# Downloading HiC
mkdir -p $HIC_DIR
aws s3 --no-sign-request cp s3://human-pangenomics/working/HPRC_PLUS/HG002/raw_data/hic/downsampled/HG002.HiC_2_NovaSeq_rep1_run2_S1_L001_R1_001.fastq.gz $HIC_DIR/
aws s3 --no-sign-request cp s3://human-pangenomics/working/HPRC_PLUS/HG002/raw_data/hic/downsampled/HG002.HiC_2_NovaSeq_rep1_run2_S1_L001_R2_001.fastq.gz $HIC_DIR/

# Downloading ASM
mkdir -p $MAIN_DIR/references
wget "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/HG002/assemblies/hg002v1.1.fasta.gz" -O ${MAIN_DIR}/references/${GENOME_ID}.ref.fna.gz
cd $MAIN_DIR/references
gzip -d ${GENOME_ID}.ref.fna.gz

# Extracting Maternal haplotype (H1)
cat ${GENOME_ID}.ref.fna | \
	perl -pe '/^>/ ? print "\n" : chomp' | \
	tail -n +2 | \
	awk '/^>/ {print $0} !/^>/ {print toupper($0)}' > \
	${GENOME_ID}.ref.fna.formated
grep -E -A1 "MATERNAL" ${GENOME_ID}.ref.fna.formated > ${GENOME_ID}.fa
rm ${GENOME_ID}.ref.fna.formated

# Preparing Juicer Template and launching it
cd $MAIN_DIR
cat ${BASE_DIR}/JUICER.sh | sed "s/ctg_id/${GENOME_ID}/g" > JUICER.sh
sbatch JUICER.sh