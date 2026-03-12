# Paper scripts

This directory contains the scripts that have been used to process the datasets and produce the figures of the paper.

> [!note] Since most of the scripts have been run on the Genobioinfo cluster, some tools that have been used were already installed and loaded using the `module load` command.

## Preprocessing
### Juicer
> [!note] This section includes scripts to download Hi-C reads (Fastq) and corresponding Assembly (Fasta).

Each `<dataset>.down.sh` download Hi-C reads and the assembly, and generates a script to run JUICER based on the [JUICER.sh template](Preprocessings/JUICER.sh).
For datasets used to benchmark Hi-C scaffolders, [scaffold2ctg.awk](Preprocessings/scaffold2ctg.awk) is used to break the DToL assembly into contigs.

### MCOOL files
Then, to run `HicExplorer`, use the [HE.sh](Preprocessings/HE.sh) after tuning the variables:
```shell
# This snippet set the ctg_file variable
SPA_dir="SPA-C" # Directory of the Git repo
for dir in *.DToL HG002.B10M HG002 CHM13; do
  cat ${SPA_dir}/paper/Preprocessings/HE.sh | sed "s/<ID>/${dir}/g" > ${dir}/HE.sh
done
```
Run each script to create the `<Genome ID>.JHE.mcool` which contains Hi-C contact matrices.

### MND file
Since `3D-DNA` and `AutoHic` requires a `merged_nodups.txt` file, use the [MND.sh](Preprocessings/MND.sh) template 
within the Juicer directory to create it:
```shell
SPA_dir="SPA-C" # Directory of the Git repo
for dir in *.DToL HG002.B10M; do
  cp ${SPA_dir}/paper/Preprocessings/MND.sh ${dir}/MND.sh
done
```

## Benchmark
### Running SPA-C 
Using the [run_SPA-C.sh](../run_SPA-C.sh) template, SPA-C is run for all datasets using:
```shell
SPA_dir="SPA-C" # Directory of the Git repo
for dir in *.DToL HG002.B10M; do
  mkdir ${dir}/SPA-C
  cat ${SPA_dir}/run_SPA-C.sh | sed "s/GenomeID/${dir}/g" > ${dir}/SPA-C/run_SPA-C.sh
done
```
Run each script to get the scaffolded FASTA.

### Running YaHS
> [!note] This script should be run in the juicer directory.

Use the [YaHS.sh](Benchmark/YaHS.sh) template to run YaHS on each dataset:
```shell
SPA_dir="SPA-C" # Directory of the Git repo
for dir in *.DToL HG002.B10M; do
  cat ${SPA_dir}/paper/Benchmark/YaHS.sh | sed "s/<ID>/${dir}/g" > ${dir}/YaHS.sh
done
```

### Running Pin_hic
> [!note] This script should be run in the juicer directory.

Use the [PINHIC.sh](Benchmark/PINHIC.sh) template (change the bin directory accordingly) to run Pin_hic:
```shell
SPA_dir="SPA-C" # Directory of the Git repo
for dir in *.DToL HG002.B10M; do
  cat ${SPA_dir}/paper/Benchmark/PINHIC.sh | sed "s/<ID>/${dir}/g" > ${dir}/PINHIC.sh
done
```

### Running 3D-DNA
> [!note] This script should be run in the juicer directory.

Use the [3DDNA.sh](Benchmark/3DDNA.sh) template to run 3D-DNA:
```shell
SPA_dir="SPA-C" # Directory of the Git repo
for dir in *.DToL HG002.B10M; do
  cat ${SPA_dir}/paper/Benchmark/3DDNA.sh | sed "s/<ID>/${dir}/g" > ${dir}/3DDNA.sh
done
```

### Running AutoHic
> [!note] This script should be run in the juicer directory.

Use the [AUTOHIC.sh](Benchmark/AUTOHIC.sh) template to run AutoHic:
```shell
SPA_dir="SPA-C" # Directory of the Git repo
for dir in *.DToL HG002.B10M; do
  cat ${SPA_dir}/paper/Benchmark/AUTOHIC.sh | sed "s/<ID>/${dir}/g" > ${dir}/AUTOHIC.sh
done
```

### Running QUAST
> [!note] This script should be run in the juicer directory.

Use the [Quast.sh](Benchmark/Quast.sh) template to compare Hi-C scaffolders output to the curated reference:
```shell
SPA_dir="SPA-C"
for dir in *.DToL HG002.B10M; do
  cat ${SPA_dir}/paper/Benchmark/Quast.sh | sed "s/<ID>/${dir}/g" > ${dir}/Quast.sh
  mkdir ${dir}/fasta.out
  ln -s ${dir}/yahs.out/yahs.out_scaffolds_final.fa ${dir}/fasta.out/YaHS.fa
  ln -s ${dir}/pinhic.out/scaffolds_final.fa ${dir}/fasta.out/Pin_hic.fa
  ln -s ${dir}/3DDNA/${dir}.FINAL.fasta ${dir}/fasta.out/3DDNA.FINAL.fa
  ln -s ${dir}/AutoHic/${dir}.FINAL.fasta ${dir}/fasta.out/AutoHic.fa
  ln -s ${dir}/SPA-C/${dir}.out.fa ${dir}/fasta.out/SPA-C.fa
done
```

## Figures
In order to reproduce figures, first start Jupyter Lab with the development environment:
```shell
apps="apptainer" # Directory of apptainer images
SPA_dir="SPA-C" # Directory of the Git

cd ${SPA_dir}/paper/Figures
apptainer exec --cleanenv --nv $apps/LENV.sif jupyter lab --port 8888
```