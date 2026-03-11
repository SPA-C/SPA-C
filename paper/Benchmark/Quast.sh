#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH -o <ID>.Quast.log
#SBATCH -J <ID>.Quast
#SBATCH --mem=164G
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

# Specific to the Genotoul-bioinfo cluster
module load bioinfo/QUAST/5.2.0
ID="<ID>"

quast.py \
    "references/${ID}.fa" \
    fasta.out/YaHS.fa \
    fasta.out/Pin_hic.fa \
    fasta.out/3DDNA.FINAL.fa \
    fasta.out/AutoHic.fa \
    fasta.out/SPA-C.fa \
    -r "references/${ID}.ref.fna" \
    -o "Quast.${ID}" \
    --threads 16 \
    --large
