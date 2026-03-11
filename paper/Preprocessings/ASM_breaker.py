#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assembly breaker

@author: Alexis Mergez
@Last modified: 2025/08/12
@version: 1.1
"""

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

version = "1.1"

#% Parsing arguments
argParser = argparse.ArgumentParser(
    description= f"Assembly breaker - {version}"
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
    "--out",
    "-o",
    dest = "out_file",
    required = True,
    help = "Path to the output file",
    type = str
    )
argParser.add_argument(
    "--size",
    dest = "ctg_size",
    default = 1000000, # 1Mb
    help = "Contig size (Default : 1000000)",
    type = int
    )
argParser.add_argument(
    "--simple-names",
    dest = "simple_names",
    help = "Simple contig names",
    action='store_true'
    )
args = argParser.parse_args()

##% Loading fasta
print(f"Parsing fasta ...")
Fasta = {
    seq_record.id: str(seq_record.seq)
    for seq_record in SeqIO.parse(args.fasta_file, "fasta")
    }
print("Done")

##% Breaking contigs
print(f"Breaking contigs...")

out_fasta = {}
conversion_table = {}

for ctg_name in Fasta.keys():
    ctg_len = len(Fasta[ctg_name])

    letter = 0 # ASCII letter for simple naming
    if ctg_len >= args.ctg_size:
        frag_length = ctg_len // (ctg_len // args.ctg_size)

        lims = list(range(0, ctg_len, frag_length))

        # Adding/Modifying outer bound to include whole contig
        lims[-1] = ctg_len-1

        for start, end in zip(lims[:-1], lims[1:]):
            if args.simple_names:
                out_fasta[f"{ctg_name}_{chr(65+letter)}"] = Fasta[ctg_name][start:end]
                conversion_table[f"{ctg_name}_{chr(65+letter)}"] = f"{ctg_name}:{start}-{end}"
                letter += 1
            else:
                out_fasta[f"{ctg_name}:{start}-{end}"] = Fasta[ctg_name][start:end]

    
##% Writing contigs
print(f"Writing broken contigs...")
with open(args.out_file, "w") as handle:
    for ctg_name in out_fasta.keys():
        handle.write(f">{ctg_name}\n")
        handle.write(out_fasta[ctg_name]+"\n")

if args.simple_names:
    output_file = args.out_file.rsplit(".", 1)[0]
    print(f"Writing conversion table to {output_file}.conversion_table.txt...")
    with open(f"{output_file}.conversion_table.txt", "w") as handle:
        for key, value in conversion_table.items():
            handle.write(f"{key}\t{value}\n")
