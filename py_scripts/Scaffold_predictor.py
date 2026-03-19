"""
Scaffold predictor script for DLScaff

@author: alexis.mergez@inrae.fr
@version: 0.8
"""

# Librairies -----------------------------------------------------------------------------------------------------------
import os
from torch.utils.data import DataLoader
from SPAC_Dataset import dataset
from models import DLScaff
import json
import torch
from tqdm import tqdm
import argparse
import numpy as np

import networkx as nx

# Fixed variables ------------------------------------------------------------------------------------------------------
input_shape = (20, 20)
latent_width = 64
batch_size = 64
n_workers = 1
version = "0.8"

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
    description= f"DLScaff chimeric contig predictor - v{version}"
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
    dest = "ds_dir",
    required = True,
    help = "Path to the HDF5 file(s). Keep the same order when using multiple resolutions",
    type = str,
    action='append',
    nargs='+'
    )
argParser.add_argument(
    "--fasta",
    "-f",
    dest = "fasta_dir",
    required = True,
    help = "Path to the assembly fasta",
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
    "--output",
    "-g",
    dest = "gfa_dir",
    required = None,
    help = "Path of the output graph (GFAv1)",
    type = str
    )
argParser.add_argument(
    "--threads",
    "-t",
    dest = "threads",
    required = None,
    help = "Number of workers",
    type = int,
    default = 1
    )
argParser.add_argument(
    "--bin-size",
    dest = "bin_size",
    required = True,
    help = "Bin size (resolution). Keep the same order as data dir.",
    action='append',
    nargs='+',
    type = int
    )
args = argParser.parse_args()

main_dir = args.main_dir
ds_dir = args.ds_dir[0]
model_weights = args.model_weights
fasta_dir = args.fasta_dir
bin_sizes = args.bin_size[0]

if args.gfa_dir is None: gfa_dir = os.path.join(main_dir, "scaffolds.gfa")
else: gfa_dir = args.gfa_dir

# Loading model --------------------------------------------------------------------------------------------------------
model = DLScaff(
    device=device,
    name="DLScaff",
    latent_width=latent_width,
    input_shape=(1, 20, 20),
    weights=model_weights
)

print(ds_dir, bin_sizes)
data = {bin_size: {"DIR": dir} for bin_size, dir in zip(bin_sizes, ds_dir)}

# Parsing fasta --------------------------------------------------------------------------------------------------------
def parse_fasta(filename):
    sequences = {}
    with open(filename, "r") as f:
        current_id = None
        current_seq = []

        for line in f:
            line = line.strip()
            if line.startswith(">"):  # New sequence
                if current_id:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split(" ")[0] # Removing ">"
                current_seq = []
            else:
                current_seq.append(line)

        # Adding the last sequence
        if current_id:
            sequences[current_id] = "".join(current_seq)

    return sequences

fasta_data = parse_fasta(fasta_dir)

# Predicting -----------------------------------------------------------------------------------------------------------
for bin_size in bin_sizes:
    print(f"Loading dataset @ {bin_size} bin size")
    data[bin_size]["DS"] = dataset(data[bin_size]["DIR"], image_size=10, bin_size=bin_size, skip_whole=True,
                                   skip_check=True)
    data[bin_size]["DS"].minmax(quantile=.95)
    data[bin_size]["DL"] = DataLoader(data[bin_size]["DS"].get_as_torch(), batch_size=batch_size, shuffle=False,
                                      num_workers=args.threads, pin_memory=(device=="cuda"))

    print(f"Predicting @ {bin_size} bin size")
    _ = model.predict(
        data[bin_size]["DL"],
        savedir=os.path.join(main_dir, f"scaff_scores_{bin_size}.json"),
        names=data[bin_size]["DS"].dataset["names"]
    )

    with open(os.path.join(main_dir, f"scaff_scores_{bin_size}.json"), "r") as handle:
        data[bin_size]["PROBS"] = json.load(handle)

    # Removing dataset in case of huge datasets
    del data[bin_size]["DS"]
    del data[bin_size]["DL"]

# Processing predictions
def start_or_end(pos):
    if pos == 0: return "Start"
    return "End"

def get_str_net(prob_net):
    str_net = {}
    for main_key, info in prob_net.items():
        str_net['|'.join(main_key)] = {}
        for second_key, prob in info.items():
            str_net['|'.join(main_key)]['|'.join(second_key)] = round(prob, 4)
    return str_net

def get_filtered_net(str_net):
    filt_net = {}
    for main_key, info in str_net.items():
        for second_key, prob in info.items():
            if prob >= 0.1:
                try:
                    filt_net[main_key][second_key] = round(prob, 4)
                except:
                    filt_net[main_key] = {second_key: round(prob, 4)}
    return filt_net

# Building the prob net(s)
for bin_size in bin_sizes:
    data[bin_size]["PROB_NET"] = {}
    data[bin_size]["STR_PROB_NET"] = {}
    data[bin_size]["FILT_PROB_NET"] = {}

    for ctg, prob in data[bin_size]["PROBS"].items():
        # Decomposing contig name
        try:
            start_part, end_part = ctg.split("|")
            start_ctg, start_pos = start_part.rsplit(':', 1)
            end_ctg, end_pos = end_part.rsplit(':', 1)
            start_pos, end_pos = start_or_end(int(start_pos)), start_or_end(int(end_pos))

        except:
            continue

        if not (end_ctg, start_ctg) in data[bin_size]["PROB_NET"].keys():
            main_key = (start_ctg, end_ctg)
            second_key = (start_pos, end_pos)
        else:
            main_key = (end_ctg, start_ctg)
            second_key = (end_pos, start_pos)

        try:
            data[bin_size]["PROB_NET"][main_key][second_key] = (prob + data[bin_size]["PROB_NET"][main_key][second_key]) / 2
        except:
            try:
                data[bin_size]["PROB_NET"][main_key][second_key] = prob
            except:
                data[bin_size]["PROB_NET"][main_key] = {second_key: prob}

    # Converting tuple key to str
    data[bin_size]["STR_PROB_NET"] = get_str_net(data[bin_size]["PROB_NET"])
    with open(os.path.join(main_dir, f"scaff_net_{bin_size}.json"), "w") as handle:
        json.dump(data[bin_size]["STR_PROB_NET"], handle, indent=2)

    # Filtered prob_net for debug purpose
    data[bin_size]["FILT_PROB_NET"] = get_filtered_net(data[bin_size]["STR_PROB_NET"])
    with open(os.path.join(main_dir, f"filtered_scaff_net_{bin_size}.json"), "w") as handle:
        json.dump(data[bin_size]["FILT_PROB_NET"], handle, indent=2)

# Merging prob net(s)
prob_net = {}
for bin_size in bin_sizes:
    for k1, v in data[bin_size]["PROB_NET"].items():
        for k2, p in v.items():
            if k1 in prob_net:
                if k2 in prob_net[k1]:
                    prob_net[k1][k2].append(p)
                else:
                    prob_net[k1][k2] = [p]
            else:
                prob_net[k1] = {k2: [p]}

for k1 in prob_net.keys():
    for k2 in prob_net[k1].keys():
        values = prob_net[k1][k2]
        prob_net[k1][k2] = np.mean(values)

str_net = get_str_net(prob_net)
with open(os.path.join(main_dir, f"scaff_net.json"), "w") as handle:
    json.dump(str_net, handle, indent=2)
filt_net = get_filtered_net(str_net)
with open(os.path.join(main_dir, f"filtered_scaff_net.json"), "w") as handle:
    json.dump(filt_net, handle, indent=2)

all_ctgs = np.unique( [k for bin_size in bin_sizes for tup in prob_net.keys() for k in tup] ) # Retrieving all corrected contigs (fragments)
uctgs = np.unique( [k.split(":")[0] for k in all_ctgs] ) # Retrieving base contig names
all_ctgs = list(all_ctgs) + [ f"{k}:0-{len(seq)}" for k, seq in fasta_data.items() if k not in uctgs ] # Adding uncorrected contigs back

# Building prediction network ------------------------------------------------------------------------------------------
uDG = nx.MultiDiGraph()
nodes = np.unique(all_ctgs).tolist()
node_attributes = {node: {
    "Chromosome": node.rsplit(":", 1)[0],
} for node in nodes}
uDG.add_nodes_from(node_attributes.items())

for (seq1, seq2), orientations in prob_net.items():
    for (orientA, orientB), weight in orientations.items():
        if orientB == "End" and orientA == "Start":
            uDG.add_edge(seq2, seq1, type=f"{orientB}-{orientA}", weight=round(weight, 4), score=str(round(weight, 4)))
        else:
            uDG.add_edge(seq1, seq2, type=f"{orientA}-{orientB}", weight=round(weight, 4), score=str(round(weight, 4)))

print("Nodes :", len(uDG.nodes()))
print("Edges :", len(uDG.edges()))

# Writing to GFA ------------------------------------------------------------------------------------------------------
orient_dict = {
    "End-Start": ("+", "+"),
    "Start-Start": ("-", "+"),
    "End-End": ("+", "-"),
    "Start-End": ("-", "-")
}

def graph_to_gfa(graph, threshold=0.2):
    gfa_lines = [
        "H\tVN:Z:1.0",
    ]

    # Adding nodes with their sequences
    for node, data in graph.nodes(data=True):

        chrom = data.get("Chromosome", 0)
        try :
            start, end = node.split(":")[-1].split("-")
            start, end = int(start), int(end)
        except :
            start, end = 0, -1

        seq = fasta_data[chrom][start:end]

        gfa_lines.append(f"S\t{node}\t{seq.upper()}\tLN:i:{len(seq)}")

    # Adding links and their weights
    for source, target, data in graph.edges(data=True):
        OrA, OrB = orient_dict[data["type"]]
        if data.get('weight', 0) >= threshold:
            gfa_lines.append(
                f"L\t{source}\t{OrA}\t{target}\t{OrB}\t0M\tFC:i:{data.get('score', 0)}")
    return "\n".join(gfa_lines)

with open(gfa_dir, "w") as handle:
    handle.write(graph_to_gfa(uDG, threshold=0))

print("Done.")