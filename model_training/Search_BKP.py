#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to search breakpoints in a dataset

@author: Alexis Mergez
@Last modified: 2025/01/20
@version: 1.0
"""

import utils
from utils import Interval
import pandas as pd
import os
from sklearn.cluster import DBSCAN
import numpy as np
import argparse
from tqdm import tqdm

version = "1.0"

#% Parsing arguments
argParser = argparse.ArgumentParser(
    description= f"Breakpoint searcher - v{version}"
    )
argParser.add_argument(
    "--paf",
    dest = "paf_file",
    required = True,
    help = "Path to the paf file",
    type = str
    )
argParser.add_argument(
    "--output",
    "-o",
    dest = "out_file",
    required = True,
    help = "Path to the output file",
    type = str
    )
argParser.add_argument(
    "--skip-clustering",
    action='store_false',
    dest = "clustering",
    help = "If added, skip clustering of breakpoints."
    )
argParser.add_argument(
    "--bkpNeighborFactor",
    dest = "bkpNF",
    default = 25,
    help = "Multiplier for the neighboring range of DBSCAN search",
    type = int
    )
argParser.add_argument(
    "--bkpMinAlignSize",
    dest = "bkpMAS",
    default = 0,
    help = "Minimum size of an alignement to be considered in base count",
    type = int
    )
argParser.add_argument(
    "--bkpDeadZone",
    dest = "bkpDZ",
    default = 100000,
    help = "Dead zones at ends of contigs where no breakpoint is allowed in base count",
    type = int
    )
argParser.add_argument(
    "--cluster",
    dest = "cluster",
    help = "Run on cluster",
    action='store_true'
    )
args = argParser.parse_args()

#% Loading PAF
print("Loading PAF file ...")

paf = pd.read_csv(
    args.paf_file, 
    delimiter="\t", 
    header = None,
    names = [
        "Q_name", "Q_len", "Q_start", "Q_end", "Strand", "T_name", 
        "T_len", "T_start", "T_end", "N_residue", "Al_len", "Quality"
        ],
    usecols = range(12)
    )

nContigs = len(paf.Q_name.unique())

## Adding alignment length
paf.loc[:,"Al_len"] = paf["Q_end"]-paf["Q_start"]

print(f"\t{len(paf.Q_name.unique())} contigs remaining (out of {nContigs})")
print("\tDone !")

#% Searching for break points
contigNames = paf.Q_name.unique().tolist()
breakpoints, rawBreakpoints = {}, {}

for i in tqdm(range(len(contigNames)), desc=f"Searching breakpoints...", disable = args.cluster):
    contig = contigNames[i]

    if args.cluster: print(f"Working on {contig}...")
    
    ## Getting a subset paf for the contig and filtering according to alignment length 
    sub_paf = paf[(paf.Q_name == contig) & (paf.Al_len >= args.bkpMAS)]

    ## Getting an Interval object for each alignment
    allIntervals = [Interval(sub_paf.loc[i, "Q_start"], sub_paf.loc[i, "Q_end"]) for i in sub_paf.index]
    
    ## Keeping intervals not included in others
    intervals = [
        allIntervals[k] 
        for k in range(len(allIntervals))
        if not allIntervals[k].isIncludedOnce(allIntervals[:k]+allIntervals[k+1:])
        ]

    ## Computing overlaps limits between intervals i.e. breakpoints
    
    overlaps = []
    for firstInterval in intervals :
        for secondInterval in intervals :

            #print(firstInterval.getLimits(), secondInterval.getLimits(), firstInterval.isSame(secondInterval), firstInterval.isOverlaped(secondInterval))

            if not firstInterval.isSame(secondInterval) and firstInterval.isOverlaped(secondInterval) :
                overlaps.append(firstInterval.getOverlap(secondInterval).getLimits())

    ## Flattening the list
    bkps = sum(overlaps, [])
    bkps.sort()
    
    ## Removing breakpoints in dead zone
    bkps = [
        k
        for k in bkps
        if k >= args.bkpDZ
        and k <= (paf[(paf.Q_name == contig)].Q_len.max()-args.bkpDZ)
        ]

    ## Removing duplicates
    bkps = list(set(bkps))
    
    if len(bkps) != 0 :
        
        ## Saving current breakpoints as 'rawBreakpoints'
        rawBreakpoints[contig] = np.copy(bkps).tolist()
        
        if args.clustering:
            bkps = np.array(bkps)

            ## Clustering breakpoints
            clusters = DBSCAN(eps = args.bkpNF*25000, min_samples=1).fit_predict(bkps.reshape(-1, 1))
            
            ## Taking the mean within clusters
            breakpoints[contig] = [
                int(np.mean(bkps[clusters == uid])) 
                for uid in np.unique(clusters)
                ]
        else : 
            breakpoints[contig] = np.copy(bkps).tolist()
            
print(f"\tDone !")

## Saving to json
utils.save_to_json(breakpoints, args.out_file)
utils.save_to_json(rawBreakpoints, f"{args.out_file}raw")
