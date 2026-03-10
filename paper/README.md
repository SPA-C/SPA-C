# Paper scripts

This directory contains the scripts that have been used to process the datasets and produce the figures of the paper.

## Preprocessing
> [!note] This section includes scripts to download Hi-C reads (Fastq) and corresponding Assembly (Fasta).

Each `<dataset>.down.sh` download Hi-C reads and the assembly, and generates a script to run JUICER based on the [JUICER.sh template](Preprocessings/JUICER.sh).
For datasets used to benchmark Hi-C scaffolders, [scaffold2ctg.awk](Preprocessings/scaffold2ctg.awk) is used to break the DToL assembly into contigs.

## Running SPA-C (benchmarking)
Using the [run_SPA-C.sh](../run_SPA-C.sh) template, SPA-C is run for all datasets using:
```shell
for dir in *.DToL HG002.B10M; do
  mkdir ${dir}/SPA-C
  cat run_SPA-C.sh | sed "s/GenomeID/${dir}/g" > ${dir}/SPA-C/run_SPA-C.sh
  #ToDo
done
```

## Benchmark