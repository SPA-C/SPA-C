#!/bin/bash
#SBATCH -t 02-00
#SBATCH --mem=192G
#SBATCH -c 16
#SBATCH -o ctg_id.Juicer.log
#SBATCH -J ctg_id.Juicer
#SBATCH --mail-type=BEGIN,END,FAIL

module load bioinfo/bwa/0.7.19
module load bioinfo/samtools/1.21
module load containers/Apptainer/1.2.5

apps="apptainer" # Path to apptainer images
SCRIPTS="juicer/CPU" # Path to Juicer CPU scripts
BASE_DIR="datasets" # Directory containing all datasets

ctg_file="ctg_id"

# Indexing
if [ ! -f "references/${ctg_file}.fa.chrom.sizes" ]; then
  echo "Indexing"
  bwa index references/${ctg_file}.fa
  samtools faidx references/${ctg_file}.fa
  cat references/${ctg_file}.fa.fai | cut -f1,2 > references/${ctg_file}.fa.chrom.sizes
fi

# Running Juicer
if [ ! -f "aligned/merged_dedup.bam" ]; then
  /usr/bin/time -v -o Juicer.${ctg_file}.time.log $SCRIPTS/juicer.sh \
    -D ${BASE_DIR}/${ctg_file} \
    -z ${BASE_DIR}/${ctg_file}/references/${ctg_file}.fa \
    -p ${BASE_DIR}/${ctg_file}/references/${ctg_file}.fa.chrom.sizes \
    -t $(nproc) -e
fi

# Printing execution time in log.
squeue -j ${SLURM_JOBID}

rm aligned/merged_sort.bam
rm -r splits