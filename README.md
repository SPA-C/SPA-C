# DLScaff

DLScaff is a CNN based tool to correct and scaffold assemblies.

## Installation
Clone this repo to your desired location :
```shell
git clone https://forge.inrae.fr/dlscaff/dlscaff.git
```

Download the apptainer image containing the environment to run the tool :
```shell
apptainer pull DLScaff.sif oras://registry.forge.inrae.fr/dlscaff/env/dlscaff:latest
```

Optionally dowload the apptainer container to preprocess your Hi-C reads :
> [!warning] ToDo

## Usage
### Preprocessing
Use Juicer or ([Juicer.gz](https://forge.inrae.fr/alexis.mergez/juicer-gz.git)) to map and preprocess your Hi-C reads.
For example: 
```shell
juicer.sh \
    -D <main_dir> \
    -z <input_assembly> \
    -p <input_assembly>.chrom.sizes \
    -t $(nproc) -e
```
> [!note]
> Make sure your input assembly has been indexed using `bwa index`.
> To generate chromosome sizes, you can use this little snippet (which uses samtools fasta index):
> ```shell
> samtools faidx <input_assembly>
> cat <input_assembly>.fai | cut -f1,2 > <input_assembly>.chrom.sizes
> ```

Using the `merged_dedup.bam` produced by Juicer, we split both R1 and R2 reads into dedicated BAMs:
```shell
samtools view -b -f 64 -@ $(nproc) merged_dedup.bam > merged_dedup.R1.bam
samtools view -b -f 128 -@ $(nproc) merged_dedup.bam > merged_dedup.R2.bam
```

Now use [HicExplorer](https://hicexplorer.readthedocs.io/en/latest/) to generate `MCOOL` files.
This MCOOL contains the HiC contact matrices that will be latter used by DLScaff.
```shell
touch restrc_cuts.bed dangling_seq.txt restrc_seq.txt

hicBuildMatrix \
  -s merged_dedup.R1.bam merged_dedup.R2.bam \
  -o <assembly>.mcool \
  --binSize 5000 1000000 \
  -cs <input_assembly>.chrom.sizes \
  --threads $(nproc) \
  --QCfolder QC_folder \
  -rs restrc_cuts.bed \
  -seq restrc_seq.txt \
  --danglingSequence dangling_seq.txt \
  --skipDuplicationCheck \
  --minMappingQuality 0
```

We provide an Apptainer container which contains both Juicer.gz and HicExplorer.
> [!warning] ToDo
> 

### Correcting and scaffolding the assembly
Modify paths in the script run.sh according to your configuration and run it.

### Outputs
> [!warning] ToDo
