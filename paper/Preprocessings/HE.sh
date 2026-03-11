#!/bin/bash  
#SBATCH -t 24:00:00
#SBATCH --mem=192G  
#SBATCH -c 4  
#SBATCH -o <ID>.HE.log
#SBATCH -J <ID>.HE
#SBATCH --mail-type=BEGIN,END,FAIL

# Specific to the Genotoul-bioinfo cluster
module load bioinfo/samtools/1.21  
module load containers/Apptainer/1.2.5  

apps="apptainer" # Path to apptainer images folder

ctg_file="<ID>"

# Splitting paired-reads into R1 and R2  
if [ ! -f "aligned/merged_dedup.R2.bam" ]; then  
  echo "Splitting R1/R2"  
  samtools view -b -f 64 -@ $(nproc) aligned/merged_dedup.bam > aligned/merged_dedup.R1.bam  
  samtools view -b -f 128 -@ $(nproc) aligned/merged_dedup.bam > aligned/merged_dedup.R2.bam  
fi  

touch restrc_cuts.bed dangling_seq.txt restrc_seq.txt  
echo "Converting to MCOOL @ Q0"  
/usr/bin/time -v -o HE.time.log $apps/HiC.sif hicBuildMatrix \
  -s aligned/merged_dedup.R1.bam aligned/merged_dedup.R2.bam \
  -o aligned/${ctg_file}.JHE.mcool \
  --binSize 1000 5000 10000 25000 100000 500000 1000000 \
  -cs references/${ctg_file}.fa.chrom.sizes \
  --threads $(nproc) \
  --QCfolder HiC_QC.Q0 \
  -rs restrc_cuts.bed \
  -seq restrc_seq.txt \
  --danglingSequence dangling_seq.txt \
  --skipDuplicationCheck \
  --minMappingQuality 0
  
# Printing execution time in log.
squeue -j ${SLURM_JOBID}

rm aligned/merged_dedup.R*.bam