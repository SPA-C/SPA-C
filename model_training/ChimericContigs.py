#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chimeric contig generator

@author: Alexis Mergez
@Last modified: 2025/11/25
@version: 1.5.5
"""

import argparse
import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

version = "1.5.5"

#% Parsing arguments
argParser = argparse.ArgumentParser(
    description= f"Chimeric contig generator - v{version}"
    )
argParser.add_argument(
    "--fasta",
    "-f",
    dest = "fasta_file",
    required = True,
    help = "Path to the FASTA file",
    type = str
    )
argParser.add_argument(
    "--paf",
    "-p",
    dest = "paf_file",
    required = True,
    help = "Path to the PAF file",
    type = str
    )
argParser.add_argument(
    "--out",
    "-o",
    dest = "out_file",
    required = True,
    help = "Path to the output file",
    type = str
    )
argParser.add_argument(
    "--seed",
    dest = "seed",
    default = None,
    help = "Random seed (Default: None)",
    type = int
    )
argParser.add_argument(
    "--repetition",
    dest = "repetition",
    default = 5,
    help = "Number of chimera to generate for a given contig pair (Default: 5)",
    type = int
    )
argParser.add_argument(
    "--size",
    dest = "ctg_size",
    default = 2000000, # 2Mb : image size of 50 with 49 bins (= 98 bin wide contigs) for sliding and 4 more for buffering contig ends, at bin size of 10000
    help = "Contig size (Default : 200000)",
    type = int
    )
argParser.add_argument(
    "--cluster",
    dest = "cluster",
    help = "Run on cluster",
    action='store_true'
    )
argParser.add_argument(
    "--intra",
    dest = "intra",
    help = "Create intra-chimeric contigs only",
    action='store_true'
    )
argParser.add_argument(
    "--simple-names",
    dest = "simple_names",
    help = "Simple contig names",
    type = str,
    default = None
    )
argParser.add_argument(
    "--inversion",
    dest = "inversion",
    help = "Generate inversions from intra-chromosome regions. Automatically enables --intra",
    action='store_true'
    )
argParser.add_argument(
    "--unbalanced-inversion",
    dest = "unbalanced_inversion",
    help = "Generate unbalanced inversions from intra-chromosome regions. Automatically enables --intra and --inversion",
    action='store_true'
    )
args = argParser.parse_args()

if args.inversion : args.intra = True
if args.unbalanced_inversion :
    args.intra = True
    args.inversion = True

#% Main
half_size = args.ctg_size//2
print(f"Chimeric contig generator - {version}")

##% Loading fasta
print(f"Parsing fasta ...")
Fasta = {
    seq_record.id: str(seq_record.seq)
    for seq_record in SeqIO.parse(args.fasta_file, "fasta")
    }
print("Done")

ctg_names = [
    ctg_name 
    for ctg_name in Fasta.keys() 
    if len(Fasta[ctg_name]) >= half_size
]

##% Generating chimeric contigs
rng = np.random.default_rng(args.seed)

##% Generating available contig pairs
rng.shuffle(ctg_names)

if not args.intra:
    # Only inter-chromosome chimeras
    ctg_pairs = [
        (ctg_names[A], ctg_names[B])
        for A in range(len(ctg_names)-1)
        for B in range(A+1, len(ctg_names))
        if A != B
    ]
else:
    # Only intra-chromosome chimeras
    ctg_pairs = [
        (ctg_names[A], ctg_names[A])
        for A in range(len(ctg_names))
    ]
rng.shuffle(ctg_pairs)

##% Listing available positions for each contig to prevent overlapping contigs
ctg_pos = {
    ctg_name: list(range(0, len(Fasta[ctg_name])-half_size, half_size))
    for ctg_name in ctg_names
}

for key in ctg_pos.keys():
    rng.shuffle(ctg_pos[key])

out = {}
paf = {
    "Q_name":[],
    "Q_len":[],
    "Q_start":[],
    "Q_end":[],
    "strand":[],
    "T_name":[]
    }

contig_uid = 1
conversion_table = {}

def reverse_complement(sequence):
    reverse_dict = {"A":"T", "T":"A", "C":"G", "G":"C", "N":"N"}
    return "".join([reverse_dict[k.upper()] for k in sequence[::-1]])

def check_distance(ctg1, pos1, ctg2, pos2, mindistance = args.ctg_size, maxdistance = 5000000):
    if ctg1 == ctg2: # We check if the gap is greater than the minimal distance (= halfsize) and smaller than the max defined length.
        return mindistance <= abs(pos1-pos2) <= maxdistance
    else:
        return True

for repetition in range(1, args.repetition+1):
    for k in tqdm(range(len(ctg_pairs)), desc=f"Generating chimeras - Round {repetition}", disable = args.cluster, unit="pairs"):
        ctg1, ctg2 = ctg_pairs[k]

        if (len(ctg_pos[ctg1]) and len(ctg_pos[ctg2])): # checking if there are elements in both positions
            ## --- Selecting both starting positions ---
            start_ctg1 = ctg_pos[ctg1].pop()

            start_ctg2 = None
            maxdistance = 5000000 if not args.inversion else half_size
            mindistance = args.ctg_size if not args.inversion else half_size

            for k in range(len(ctg_pos[ctg2])):
                if check_distance(ctg1, start_ctg1, ctg2, ctg_pos[ctg2][k], maxdistance=maxdistance, mindistance=mindistance): # If this second position is in the desired range
                    start_ctg2 = ctg_pos[ctg2].pop(k)
                    break

            if start_ctg2 is None: continue # If no positions were adequate, we skip the position start_ctg1

            ## --- Naming and handling inversions ---
            # Reversing ctg2
            if args.inversion:
                # Shorter contig if unbalanced inversion
                end2 = start_ctg2+half_size if not args.unbalanced_inversion else start_ctg2+int((rng.integers(3, 11, size=1)/10)*half_size)

                if args.simple_names is None:
                    chimera_name = f"{ctg1}:{start_ctg1}-{start_ctg1+half_size}|{ctg2}:{end2}-{start_ctg2}"
                else:
                    chimera_name = f"{args.simple_names}_{contig_uid}"
                    conversion_table[chimera_name] = f"{ctg1}:{start_ctg1}-{start_ctg1+half_size}|{ctg2}:{end2}-{start_ctg2}"
                    contig_uid += 1

                out[chimera_name] = Fasta[ctg1][start_ctg1:start_ctg1 + half_size] + reverse_complement(Fasta[ctg2][start_ctg2:end2])

            # No reversing
            else :
                if args.simple_names is None:
                    chimera_name = f"{ctg1}:{start_ctg1}-{start_ctg1+half_size}|{ctg2}:{start_ctg2}-{start_ctg2+half_size}"
                else:
                    chimera_name = f"{args.simple_names}_{contig_uid}"
                    conversion_table[chimera_name] = f"{ctg1}:{start_ctg1}-{start_ctg1+half_size}|{ctg2}:{start_ctg2}-{start_ctg2+half_size}"
                    contig_uid += 1

                out[chimera_name] = Fasta[ctg1][start_ctg1:start_ctg1 + half_size] + Fasta[ctg2][start_ctg2:start_ctg2 + half_size]

            ### Writing PAF lines
            #### Left side (ctg1)
            paf["strand"].append(0)
            paf["Q_name"].append(chimera_name)
            paf["Q_len"].append(args.ctg_size)
            paf["Q_start"].append(0)
            paf["Q_end"].append(half_size)
            paf["T_name"].append(ctg1)

            #### Right side (ctg2)
            paf["strand"].append(0)
            paf["Q_name"].append(chimera_name)
            paf["Q_len"].append(args.ctg_size)
            paf["Q_start"].append(half_size)
            paf["Q_end"].append(args.ctg_size)
            paf["T_name"].append(ctg2)
        
        else :
            pass

    print(f"\tRound {repetition} : {len(out)} chimeric contigs in total, {sum([len(ctg_pos[ctg_name]) for ctg_name in ctg_names])} positions left.")

##% Writing to output
print(f"Writing chimeras to {args.out_file}...")
with open(args.out_file, "w") as handle:
    for chimera in out.keys():
        handle.write(f">{chimera}\n")
        handle.write(out[chimera]+"\n")
print("Done")

##% Writing PAF file
print(f"Writing alignement to {args.paf_file}...")
for i in range(12-len(paf)):
    paf[f"placeholder_{i}"] = [0]*len(paf["Q_name"])

df = pd.DataFrame(data=paf)
df.to_csv(args.paf_file, header=False, index=False, sep='\t')

##% Writing conversion table in case
if args.simple_names is not None:
    output_file = args.paf_file.rsplit(".", 1)[0]
    print(f"Writing conversion table to {output_file}.conversion_table.txt...")
    with open(f"{output_file}.conversion_table.txt", "w") as handle:
        for key, value in conversion_table.items():
            handle.write(f"{key}\t{value}\n")

