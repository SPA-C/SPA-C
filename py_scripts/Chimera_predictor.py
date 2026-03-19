"""
Chimera predictor script for SPA-C

@author: alexis.mergez@inrae.fr
@version: 0.11.0
"""

# Librairies -----------------------------------------------------------------------------------------------------------
import os
from torch.utils.data import DataLoader
from SPAC_Dataset import dataset
from models import SPAC
import json
from tqdm import tqdm
import argparse
import torch
import numpy as np

# Fixed variables ------------------------------------------------------------------------------------------------------
input_shape = (20, 20)
latent_width = 64
batch_size = 64
n_workers = 1
smoothing_window = 20 # Arbitrary set to window size
version = "0.11.0"

# Optimisation de CuDNN
torch.backends.cudnn.benchmark = True

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Variables ------------------------------------------------------------------------------------------------------------
argParser = argparse.ArgumentParser(
    description= f"SPA-C chimeric contig predictor - v{version}"
    )
argParser.add_argument(
    "--dir",
    dest = "main_dir",
    default = "./",
    help = "Main path",
    type = str
    )
argParser.add_argument(
    "--data",
    "-d",
    dest = "ds_dir",
    required = True,
    help = "Path to the HDF5 file",
    type = str
    )
argParser.add_argument(
    "--weights",
    "-w",
    dest = "model_weights",
    required = True,
    help = "Path to the model weights",
    type = str
    )
argParser.add_argument(
    "--chrom-sizes",
    dest = "chrom_sizes",
    required = True,
    help = "Path to the chrom sizes",
    type = str
    )
argParser.add_argument(
    "--longdust",
    "-l",
    dest = "ld_path",
    required = True,
    help = "Path to longdust predictions",
    type = str
    )
argParser.add_argument(
    "--threshold",
    dest = "threshold",
    required = True,
    help = "Prediction threshold",
    type = float,
    default = 0.5
    )
argParser.add_argument(
    "--smoothing",
    dest = "smoothing",
    help = "Smooth predictions",
    action='store_true'
    )
argParser.add_argument(
    "--bin-size",
    dest = "bin_size",
    required = False,
    help = "Bin size (resolution)",
    type = int,
    default = 5000
    )
argParser.add_argument(
    "--threads",
    "-t",
    dest = "threads",
    required = False,
    help = "Number of workers",
    type = int,
    default = 1
    )
args = argParser.parse_args()

main_dir = args.main_dir
ds_dir = args.ds_dir
model_weights = args.model_weights
bin_size = args.bin_size
n_workers = args.threads

# Loading model --------------------------------------------------------------------------------------------------------
model = SPAC(
    device=device,
    name="SPA-C",
    latent_width=latent_width,
    input_shape=(1, 20, 20),
    weights=model_weights
)

# Loading dataset ------------------------------------------------------------------------------------------------------
ds = dataset(ds_dir, image_size = 10, bin_size = bin_size, skip_whole=True, skip_check=True)
ds.minmax(quantile=.95)
ds_dataloader = DataLoader(ds.get_as_torch(), batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)

# Predicting -----------------------------------------------------------------------------------------------------------
probs, _ = model.predict(
    ds_dataloader,
    savedir=os.path.join(main_dir, f"chim_scores.json"),
    names=ds.dataset["names"]
)

with open(os.path.join(main_dir, f"chim_scores.json"), "r") as handle:
    probs = json.load(handle)

intra_probs = {}

for key, value in probs.items():
    # Intra contig prediction
    # Decomposing the key
    chrom_name, bin_number, corner_id = key.rsplit("_", 2)
    bin_number = int(bin_number)

    # Adding chromosome to intra_probs if not already
    if not chrom_name in intra_probs.keys():
        intra_probs[chrom_name] = {}

    # Adding bin_number if not already
    if not bin_number in intra_probs[chrom_name].keys():
        intra_probs[chrom_name][bin_number] = {"C2": None, "C3": None}

    # Adding the value
    intra_probs[chrom_name][bin_number][corner_id] = value

# Parsing Longdust prediction
ld_regions = {}
with open(args.ld_path, "r") as handle:
    for l in handle.readlines():
        lines = l.strip().split("\t")
        if int(lines[2])-int(lines[1]) >= 2*bin_size:
            start, end = int(lines[1])//bin_size, int(lines[2])//bin_size
            try:
                ld_regions[lines[0]].append((start, end))
            except:
                ld_regions[lines[0]] = [(start, end)]

def is_in_range(x, regions, offset=5): # accounting for the fact that the image could contain the repetition pattern even if on its side
    for start, end in regions:
        if start-offset <= x <= end+offset:
            return 1
    return 0

def reading_chrom_sizes(path):
    chrom_sizes = {}
    with open(path, 'r') as handle:
        for line in handle:
            ctg_name, length = line.rstrip().split("\t")
            chrom_sizes[ctg_name] = int(length)
    return chrom_sizes

chrom_sizes = reading_chrom_sizes(args.chrom_sizes)

# Splitting chimeric contigs -------------------------------------------------------------------------------------------
non_chimeric_ctgs = []  # Storing a list of contigs
chimeric_ctgs = []
all_ctgs = []
bin_of_interest = []
cuts = []
s_probs = {}
m_probs = {}
ld_curves = {}

def smoothing_predictions(x, ctg_probs, window=smoothing_window):
    """
    Smoothing 'algorithm' using np.convolve to compute the rolling average with a size of <smoothing window>.
    If the contig is smaller than the smoothing window, only the mean is computed.
    """

    mean_probs = np.mean([
        [ctg_probs[k]["C2"] for k in x],
        [ctg_probs[k]["C3"] for k in x]
    ], axis=0)

    if len([ctg_probs[k]["C2"] for k in x]) >= window:
        smoothed_probs = np.convolve(
            mean_probs,
            np.ones(window) / window,
            mode="same"
        )
    else:
        smoothed_probs = mean_probs

    return mean_probs, smoothed_probs

def compute_range(x, probs, threshold=args.threshold):
    """
    Two cases :
    - The size of the chimeric region is greater than the minimal prediction size for scaffolding -> Considered as a new contig.
    - Else -> We take the mean position as breakpoint for the whole contig.
    """

    in_keep = False
    current_range = None
    current_bad_range = None
    current_part_id = 1
    _ctg_parts = {}

    for bin_id, pred in zip(x, probs):

        if (pred >= threshold or curve[bin_id - 10] == 1):  # We keep this bin

            if in_keep == False:  # We were not in a keeping zone hence we create one
                current_range = [bin_id, bin_id]
                in_keep = True

                if current_bad_range is not None:  # We were in a bad range
                    # Updating the last bin
                    current_bad_range[1] = bin_id

                    # Updating the contig partition
                    _ctg_parts[current_part_id] = {
                        "chimeric": True,
                        "range": [current_bad_range[0], current_bad_range[1]]
                    }

                    current_part_id += 1
                    current_bad_range = None

            else:  # We were already in a keeping zone, so we update the last element of the list
                current_range[-1] = bin_id

        if (pred < threshold and curve[bin_id - 10] == 0) or bin_id == max(
                x):  # We exit the keeping region or we are at the end anyway
            in_keep = False

            if current_range is not None:  # We were in a keeping zone, so we exit it
                # Updating last bin_id to the range
                current_range[1] = bin_id

                # Updating the contig partition
                _ctg_parts[current_part_id] = {
                    "chimeric": False,
                    "range": [current_range[0], current_range[1]]
                }

                current_part_id += 1
                current_range = None

            if current_bad_range is None:
                current_bad_range = [bin_id, bin_id]
            else:
                current_bad_range[-1] = bin_id

    return _ctg_parts

ctg_parts = {}
filtered_ctg_parts = {}
chrom_ranges = {} # Stores the start and end bin of each chromosome/contig. They are left and right shifted.

for chrom in tqdm(sorted(list(intra_probs.keys())), desc=f"Cutting chromosomes...", unit="ctg"):
    # Adding current contig entry into ctg_parts
    ctg_parts[chrom] = {}

    # Ordering bin number for the given chromosome
    x = sorted(list(intra_probs[chrom].keys()))
    chrom_ranges[chrom] = (x[0], x[-1]) # Left and right shifted
    #print(chrom_ranges[chrom]) # DEBUG

    # Computing smoothed probability curve
    mean_probs, smoothed_probs = smoothing_predictions(x, intra_probs[chrom], smoothing_window)

    # Storing smoothed probabilities
    m_probs[chrom] = {
        "x": x,
        "probs": mean_probs
    }
    s_probs[chrom] = {
        "x": x,
        "probs": smoothed_probs
    }
    try:
        curve = np.array([
            is_in_range(pos, ld_regions[chrom])
            for pos in x
        ])
    except:
        curve = np.array([
            0
            for pos in x
        ])

    ld_curves[chrom] = {
        "x": x,
        "probs": curve
    }

    # Computing ranges of keep parts
    if args.smoothing:
        ctg_parts[chrom] = compute_range(x, smoothed_probs, threshold=args.threshold)
    else:
        ctg_parts[chrom] = compute_range(x, mean_probs, threshold=args.threshold)

with open(os.path.join(main_dir, f"ctgs_parts.json"), "w") as handle:
    json.dump(ctg_parts, handle, indent=2)

filtered_ctg_parts = ctg_parts.copy()
for chrom in tqdm(sorted(list(intra_probs.keys())), desc=f"Cutting chromosomes...", unit="ctg"):
    # Finalizing chimeric detection
    k = 0
    while k < len(list(filtered_ctg_parts[chrom].keys())) - 1:  # While we are still in the contig
        # We have to dynamically parse the contig sections because we are removing some along the way
        part_ids = list(filtered_ctg_parts[chrom].keys())
        start, end = filtered_ctg_parts[chrom][part_ids[k]]["range"]

        if end - start < 20 and len(part_ids) > 1:
            # In this case, we split the current part between the previous and next part if possible.
            if k not in [0, len(part_ids) - 1]:
                filtered_ctg_parts[chrom][part_ids[k - 1]]["range"][1] = start + (end - start) // 2
                filtered_ctg_parts[chrom][part_ids[k + 1]]["range"][0] = start + (end - start) // 2

            elif k == 0:  # Removing this part entirely because it is a tip and would produce unusable small contigs
                filtered_ctg_parts[chrom][part_ids[k + 1]]["range"][0] = start

            elif k == len(part_ids) - 1:  # Removing this part entirely because it is a tip and would produce unusable small contigs
                filtered_ctg_parts[chrom][part_ids[k - 1]]["range"][1] = end

            del filtered_ctg_parts[chrom][part_ids[k]]

        else:  # We do nothing and we continue
            k += 1

# Outputting
with open(os.path.join(main_dir, f"nochim_ctgs.txt"), "w") as handle:
    for chrom, parts in filtered_ctg_parts.items():
        for part_info in parts.values():
            # Shifting Start and End positions if we are at the Start or End of the contig.
            _start = part_info['range'][0] if part_info['range'][0] != chrom_ranges[chrom][0] else 0
            _end = part_info['range'][1] if part_info['range'][1] != chrom_ranges[chrom][1] else chrom_sizes[chrom]/bin_size

            handle.write(f"{chrom}:{int(_start*bin_size)}-{int(_end*bin_size)}\n")

with open(os.path.join(main_dir, f"ctgs_filtered_parts.json"), "w") as handle:
    json.dump(filtered_ctg_parts, handle, indent=2)

print("Done.")