#!/bin/bash
# Please note that outputs of this script will be written in the same directory as this script.
# If you run DLScaff in the Juicer folder, default variables expect the tool to be run in a specific folder under the
# Juicer directory. For example: <Genome ID>/DLScaff

export SINGULARITYENV_MAMBA_ROOT_PREFIX="$TMPDIR/mamba"

# Variables
apps="" # Path to apptainer images folder
scripts="" # Path to py_script folder from the DLScaff repo
weights="" # Path to model's weights (.pth file)
ID="" # Genome ID (should be <name> given the input assembly <name>.fa)
FA="../references/${ID}.fa" # Path to the input FASTA (default is when run in a dedicated folder inside Juicer's folder)
MCOOL="../aligned/${ID}.JHE.mcool" # Path to MCOOL file (default is when run in a dedicated folder inside Juicer's folder)
LD="../references/${ID}.longdust" # Path to Longdust predictions (default is when run in a dedicated folder inside Juicer's folder)

CS="${FA}.chrom.sizes" # Path to chromosome size

mkdir tmp

# Main
# Creating Intra contig dataset
echo "[DLScaff::main] Creating prediction dataset ..."
apptainer exec $apps/dlscaff.sif python -u ${scripts}/Cool2IntraM_FullMat.py \
  --cool ${MCOOL} \
  --image-size 10 \
  --bin-size 5000 \
  --output ${ID}.intra.hdf5

## Predicting low complexity regions
if [ ! -f "${LD}" ]; then
	echo "[DLScaff::main] Predicting low complexity regions with Longdust ..."
	/usr/bin/time -v -o LD.time.log apptainer exec $apps/dlscaff.sif longdust \
	  ${FA} > ${LD}
else
	echo "[DLScaff::main] Reusing Longdust predictions ..."
fi

# Searching and correcting chimeric contigs
echo "[DLScaff::main] Correcting contigs ..."
/usr/bin/time -v -o Chim.pred.time.log apptainer exec --nv $apps/dlscaff.sif python -u ${scripts}/Chimera_predictor.py \
  --dir ./ \
  --data ${ID}.intra.hdf5 \
  --weights $weights \
  --threshold 0.05 \
  --chrom-sizes ${CS} \
  --longdust ${LD}

# Creating scaffolding dataset
echo "[DLScaff::main] Creating scaffolding dataset ..."
apptainer exec $apps/HiC.sif python -u ${scripts}/Cool2InterM_FullMat.py \
  --cool ${MCOOL} \
  --image-size 10 \
  --bin-size 5000 \
  --inference \
  --threads $(nproc) \
  --tmp-dir tmp \
  --chrom-sizes ${CS} \
  --contigs nochim_ctgs.txt \
  --output ${ID}.inter.5k.hdf5

apptainer exec $apps/HiC.sif python -u ${scripts}/Cool2InterM_FullMat.py \
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
echo "[DLScaff::main] Predicting contig linkage ..."
apptainer exec --nv $apps/dlscaff.sif python -u ${scripts}/Scaffold_predictor.py \
  --dir ./ \
  --data ${ID}.inter.5k.hdf5 ${ID}.inter.25k.hdf5 \
  --weights $weights \
  --fasta ${FA} \
  --bin-size 5000 25000 \
  --output ${ID}.gfa

# Running YaHS scaffolding algorithm (YaHS++)
echo "[DLScaff::main] Running YaHS scaffolding algorithm ..."
apptainer exec --nv $apps/dlscaff.sif yahspp \
  -i ${ID}.gfa \
  -o ${ID}.out.fa \
  -O ${ID}.out.agp \
  -m 10000 \
  -h 0.9

echo "[DLScaff::main] Done."