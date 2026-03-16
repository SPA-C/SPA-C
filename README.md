# SPA-C

SPA-C is a CNN based tool to correct and scaffold assemblies.

## Installation
Clone this repo to your desired location :
```shell
git clone https://github.com/SPA-C/SPA-C.git
```

Download the apptainer image containing the environment to run the tool (see its [Repo](https://forge.inrae.fr/SPA-C/env)) :
```shell
apptainer pull SPA-C.sif oras://registry.forge.inrae.fr/spa-c/env/spac:latest
```

Optionally, download the apptainer container to process your Hi-C reads into MCOOL files:
```shell
apptainer pull HiC.sif oras://registry.forge.inrae.fr/spa-c/hic-env/hic-env:latest
```

## Usage
### Preprocessing
Use [Juicer](https://github.com/aidenlab/juicer) or ([Juicer.gz](https://forge.inrae.fr/alexis.mergez/juicer-gz.git)) to map and preprocess your Hi-C 
reads.
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
This MCOOL contains the HiC contact matrices that will be latter used by SPA-C.
```shell
touch restrc_cuts.bed dangling_seq.txt restrc_seq.txt

hicBuildMatrix \
  -s merged_dedup.R1.bam merged_dedup.R2.bam \
  -o <assembly>.mcool \
  --binSize 1000 5000 10000 25000 100000 500000 1000000 \
  -cs <input_assembly>.chrom.sizes \
  --threads $(nproc) \
  --QCfolder QC_folder \
  -rs restrc_cuts.bed \
  -seq restrc_seq.txt \
  --danglingSequence dangling_seq.txt \
  --skipDuplicationCheck \
  --minMappingQuality 0
```

> [!note]
> If you use our Apptainer image, run: `apptainer exec HiC.sif hicBuildMatrix [...]` instead of `hicBuildMatrix [...]`.

### Correcting and scaffolding an assembly
Use the script [run_SPA-C.sh](run_SPA-C.sh) according to run SPA-C. Model's weights are located in the [weights](weights) folder.
```shell
Usage: run_SPA-C.sh [options]

Options:
  -A <dir>                Directory of apptainer images
  -S <dir>                Directory of SPA-C's scripts (py_scripts from the repo)
  -W <file>.pth           Model's weights
  -I <name>               Name of the genome (<name> in <name>.fa)
  -F <file>.fa            Assembly to scaffold (FASTA)
  -M <file>.mcool         Hi-C matrices from HicExplorer (MCOOL)
  -L <file>.longdust      Path to longdust predictions (will be generated if not found)
  -C <file>.chrom.sizes   Path to the chrom sizes file (will be generated if not found)
```

### Outputs
SPA-C generates several outputs that can be used to understand the choices that have been made:
- Misjoin detection in input's contigs:
  - `chim_scores.json` gives the raw predictions of the model along the input contigs.
  - `ctgs_parts.json` gives the raw decomposition of each input contigs into chimeric or non-chimeric regions.
  - `ctgs_filtered_parts.json` gives the decomposition of each input contigs after having processed the chimeric 
    regions into dedicated contigs or split chimeric regions in half.
  - `nochim_ctgs.txt` lists all non-chimeric contigs as regions of the input's contigs.
- Scaffolding of corrected contigs:
  - `scaff_scores_<bin resolution>.json` gives the raw predictions of the model between contig's ends, for a given 
    _bin resolution_.
  - `scaff_net_<bin resolution>.json` gives the predictions of the model grouped by contig pairs and labeled 
    following the orientation, for a given _bin resolution_.
  - `scaff_net.json` gives the predictions of the model, averaged between available _bin resolutions_ and grouped by 
    contig pairs.
  - `filtered_scaff_net.json` and `filtered_scaff_net_<bin resolution>.json` are versions where low probability 
    linkages were removed for easier understanding. **They serves for debug only and are not used in the pipeline.**
  - `<ID>.gfa` is the scaffolding graph passed to [YaHS++](https://forge.inrae.fr/SPA-C/yahspp).
  - `<ID>.out.agp` is the description of scaffolds arrangement made by [YaHS++](https://forge.inrae.fr/SPA-C/yahspp).
  - `<ID>.out.fa` is the final scaffolded assembly.
- Datasets:
  - All HDF5 files are datasets used by SPA-C's model for prediction.