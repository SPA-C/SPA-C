# SPA-C training instructions
> [!notes] To run the following scripts, we assume that base CHM13 and HG002 MCOOL files were generated (see [MCOOL generation](../paper/README.md#mcool-files)).

## Generating datasets
### CHM13
First, we create fasta files containing chimera contigs.
```shell
apps="apptainer" # Path to apptainer images folder
SPA_dir="SPA-C" # Directory of the Git repo
BASE_DIR="datasets"
DS="CHM13"
dir="${BASE_DIR}/${DS}" # Directory of CHM13 dataset

# Inter only
mkdir -p ${BASE_DIR}/${DS}.C2A/references
$apps/HiC.sif python -u ${SPA_dir}/model_training/ChimericContigs.py \
  --fasta ${dir}/references/${DS}.fa \
  --out ${BASE_DIR}/${DS}.C2A/references/${DS}.C2A.fa \
  --seed 42 \
  --paf ${BASE_DIR}/${DS}.C2A/references/${DS}.C2A.paf \
  --repetition 2000 \
  --size 200000 \
  --simple-names "C2A"

# Intra only
mkdir -p ${BASE_DIR}/${DS}.C2B/references
$apps/HiC.sif python -u ${SPA_dir}/model_training/ChimericContigs.py \
  --fasta ${dir}/references/${DS}.fa \
  --out ${BASE_DIR}/${DS}.C2B/references/${DS}.C2B.fa \
  --seed 42 \
  --paf ${BASE_DIR}/${DS}.C2B/references/${DS}.C2B.paf \
  --repetition 2500 \
  --size 200000 \
  --intra \
  --simple-names "C2B"

# Inversions only
mkdir -p ${BASE_DIR}/${DS}.C2C/references
$apps/HiC.sif python -u ${SPA_dir}/model_training/ChimericContigs.py \
  --fasta ${dir}/references/${DS}.fa \
  --out ${BASE_DIR}/${DS}.C2C/references/${DS}.C2C.fa \
  --seed 42 \
  --paf ${BASE_DIR}/${DS}.C2C/references/${DS}.C2C.paf \
  --repetition 2500 \
  --size 200000 \
  --inversion \
  --simple-names "C2C"
```

Run JUICER and HicExplorer using both templates:
```shell
BASE_DIR="datasets"
SPA_dir="SPA-C" # Directory of the Git repo
DS="CHM13"
cd $BASE_DIR

for dir in ${DS}.C*; do
  # Juicer
  cat ${SPA_dir}/paper/Preprocessings/JUICER.sh | sed "s/ctg_id/${dir}/g" > JUICER.sh
  # HicExplorer
  cat ${SPA_dir}/paper/Preprocessings/HE.sh | sed "s/<ID>/${dir}/g" > HE.sh
done
```

Then, we create the SPA-C datasets:
```shell
BASE_DIR="datasets"
apps="apptainer" # Path to apptainer images folder
SPA_dir="SPA-C" # Directory of the Git repo
DS="CHM13"
cd $BASE_DIR

for dir in ${DS}.C2*; do
  # "Searching" breakpoints in chimera 
  $apps/HiC.sif python -u ${SPA_dir}/model_training/Search_BKP.py \
    --paf ${dir}/references/${dir}.paf \
    --output ${dir}/references/${dir}.bkp \
    --skip-clustering \
    --bkpDeadZone 0
  
  # Creating the dataset
  mkdir ${dir}/SPA-C_ds
  $apps/HiC.sif python -u ${SPA_dir}/py_scripts/Cool2IntraM_FullMat.py \
    --cool ${dir}/aligned/${dir}.JHE.mcool \
    --image-size 10 \
    --bin-size 5000 \
    --breakpoint ${dir}/references/${dir}.bkp \
    --only-negatives \
    --output ${dir}/SPA-C_ds/${dir}.IntraFM.5K10S_Q0.hdf5 \
    --name-table ${dir}/references/${dir}.conversion_table.txt \
    --padding "-1"
done

dir="${DS}"
mkdir -p ${dir}/SPA-C_ds/tmp
$apps/HiC.sif python -u ${SPA_dir}/py_scripts/Cool2IntraM_FullMat.py \
  --cool ${dir}/aligned/${dir}.JHE.mcool \
  --image-size 10 \
  --bin-size 5000 \
  --output ${dir}/SPA-C_ds/${dir}.IntraFM.5K10S_Q0.hdf5

# Inter
$apps/HiC.sif python -u ${SPA_dir}/py_scripts/Cool2InterM_FullMat.py \
  --cool ${dir}/aligned/${dir}.JHE.mcool \
  --image-size 10 \
  --bin-size 5000 \
  --steps 5 \
  --threads 8 \
  --output ${dir}/SPA-C_ds/${dir}.InterFM.5K10S_Q0.hdf5 \
  --tmp-dir ${dir}/SPA-C_ds/tmp
```

#### Downsampling inter-chromosome matrices
Using the following python code, we subsample the inter-chromosome subset.
Start the python environment using: 
```shell
apps="apptainer"
SPA_dir="SPA-C"

cd SPA_dir/py_scripts
apptainer exec --cleanenv $apps/SPA-C.sif python
```
And then run these python commands:
```python
dir="datasets/CHM13/SPA-C_ds"

import os
from SPAC_Dataset import dataset
inter = dataset(os.path.join(dir, "CHM13.InterFM.5K10S_Q0.hdf5"), bin_size=5000, image_size=10, skip_check=True)
inter.equalize(clip_value=1e4)
inter.save(savedir=os.path.join(dir, "CHM13.InterFM.5K10S_Q0.EQ.hdf5"))
```

#### Merging subsets
```shell
apps="apptainer"
SPA_dir="SPA-C"
dir="datasets"
DS="CHM13"

cd $dir
mkdir ${DS}.DS
apptainer exec --cleanenv $apps/SPA-C.sif python ${SPA_dir}/model_training/Multi2One.py \
    --dataset ${DS}/SPA-C_ds/${DS}.IntraFM.5K10S_Q0.hdf5 \
    ${DS}.C2A/SPA-C_ds/${DS}.C2A.IntraFM.5K10S_Q0.hdf5 \
    ${DS}.C2B/SPA-C_ds/${DS}.C2B.IntraFM.5K10S_Q0.hdf5 \
    ${DS}.C2C/SPA-C_ds/${DS}.C2C.IntraFM.5K10S_Q0.hdf5 \
    ${DS}/SPA-C_ds/${DS}.IntraFM.5K10S_Q0.EQ.hdf5 \
    --output ${DS}.DS/${DS}.DS_5K10S_Q0.hdf5 \
    --name ${DS}.DS_5K10S_Q0.raw
```

### HG002
Replace the `DS` variable to `HG002.H1` in each [CHM13](#chm13) code snippets. 

## Training and testing the model
The model can be trained using the [SPA-C_training](SPA_training.ipynb) notebook.
#ToDo