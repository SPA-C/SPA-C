#!/bin/bash
# Please note that outputs of this script will be written in the same directory as this script.
# If you run SPA-C in the Juicer folder, default variables expect the tool to be run in a specific folder under the
# Juicer directory. For example: <Genome ID>/SPA-C

export SINGULARITYENV_MAMBA_ROOT_PREFIX="$TMPDIR/mamba"

# Variables
apps="" # Path to apptainer images folder
scripts="" # Path to py_script folder from the SPA-C repo
weights="" # Path to model's weights (.pth file)
ID="GenomeID" # Genome ID (should be <name> given the input assembly <name>.fa)
FA="../references/${ID}.fa" # Path to the input FASTA (default is when run in a dedicated folder inside Juicer's folder)
MCOOL="../aligned/${ID}.JHE.mcool" # Path to MCOOL file (default is when run in a dedicated folder inside Juicer's folder)
LD="../references/${ID}.longdust" # Path to Longdust predictions (default is when run in a dedicated folder inside Juicer's folder)

CS="${FA}.chrom.sizes" # Path to chromosome size

# USAGE
usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -A <dir>                Directory of apptainer images
  -S <dir>                Directory of SPA-C's scripts (py_scripts from the repo)
  -W <file>.pth           Model's weights
  -I <name>               Name of the genome (<name> in <name>.fa)
  -F <file>.fa            Assembly to scaffold (FASTA)
  -M <file>.mcool         Hi-C matrices from HicExplorer (MCOOL)
  -L <file>.longdust      Path to longdust predictions (will be generated if not found)
  -C <file>.chrom.sizes   Path to the chrom sizes file

EOF
}

while getopts "A:S:W:I:F:M:L:C:" option; do
  case "$option" in
    A) apps="$OPTARG";;
    S) scripts="$OPTARG";;
    W) weights="$OPTARG";;
    I) ID="$OPTARG";;
    F) FA="$OPTARG";;
    M) MCOOL="$OPTARG";;
    L) LD="$OPTARG";;
    C) CS="$OPTARG";;
    \?) usage; exit 1;;
  esac
done


mkdir tmp

# Main
# Creating Intra contig dataset
echo "[SPA-C::main] Creating prediction dataset ..."
apptainer exec $apps/SPA-C.sif python -u ${scripts}/Cool2IntraM_FullMat.py \
  --cool ${MCOOL} \
  --image-size 10 \
  --bin-size 5000 \
  --output ${ID}.intra.hdf5

## Predicting low complexity regions
if [ ! -f "${LD}" ]; then
	echo "[SPA-C::main] Predicting low complexity regions with Longdust ..."
	apptainer exec $apps/SPA-C.sif longdust \
	  ${FA} > ${LD}
else
	echo "[SPA-C::main] Reusing Longdust predictions ..."
fi

# Searching and correcting chimeric contigs
echo "[SPA-C::main] Correcting contigs ..."
apptainer exec --nv $apps/SPA-C.sif python -u ${scripts}/Chimera_predictor.py \
  --dir ./ \
  --data ${ID}.intra.hdf5 \
  --weights $weights \
  --threshold 0.05 \
  --chrom-sizes ${CS} \
  --longdust ${LD}

# Creating scaffolding dataset
echo "[SPA-C::main] Creating scaffolding dataset ..."
apptainer exec $apps/SPA-C.sif python -u ${scripts}/Cool2InterM_FullMat.py \
  --cool ${MCOOL} \
  --image-size 10 \
  --bin-size 5000 \
  --inference \
  --threads $(nproc) \
  --tmp-dir tmp \
  --chrom-sizes ${CS} \
  --contigs nochim_ctgs.txt \
  --output ${ID}.inter.5k.hdf5

apptainer exec $apps/SPA-C.sif python -u ${scripts}/Cool2InterM_FullMat.py \
  --cool ${MCOOL} \
  --image-size 10 \
  --bin-size 25000 \
  --inference \
  --threads $(nproc) \
  --tmp-dir tmp \
  --chrom-sizes ${CS} \
  --contigs nochim_ctgs.txt \
  --output ${ID}.inter.25k.hdf5

# Predicting linkage between contigs
echo "[SPA-C::main] Predicting contig linkage ..."
apptainer exec --nv $apps/SPA-C.sif python -u ${scripts}/Scaffold_predictor.py \
  --dir ./ \
  --data ${ID}.inter.5k.hdf5 ${ID}.inter.25k.hdf5 \
  --weights $weights \
  --fasta ${FA} \
  --bin-size 5000 25000 \
  --output ${ID}.gfa

# Running YaHS scaffolding algorithm (YaHS++)
echo "[SPA-C::main] Running YaHS scaffolding algorithm ..."
apptainer exec --nv $apps/SPA-C.sif yahspp \
  -i ${ID}.gfa \
  -o ${ID}.out.fa \
  -O ${ID}.out.agp \
  -m 10000 \
  -h 0.9

echo "[SPA-C::main] Done."