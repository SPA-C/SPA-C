#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi2One tool for merging several SPA-C datasets

@author: Alexis Mergez
@version: 4.5
"""

import h5py
import numpy as np
import argparse
import datetime, time
import os

version = "4.5"

#%% Parsing arguments
argParser = argparse.ArgumentParser(
    description= f"Multi2One: tool for merging several SPA-C datasets - v{version}"
    )
argParser.add_argument(
    "--dataset",
    nargs = "+",
    dest = "datasetPath",
    required = True,
    help = "Path to datasets",
    type = str
    )
argParser.add_argument(
    "--output",
    required = True,
    type = str,
    dest = "outputDir",
    help = "Output path"
    )
argParser.add_argument(
    "--name",
    required = True,
    type = str,
    dest = "dataset_name",
    help = "Name of the dataset"
    )
args = argParser.parse_args()

for path in args.datasetPath:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Unable to find {path}")

datasetDir = os.path.dirname(args.outputDir)    
if not os.path.isdir(datasetDir):
    os.makedirs(datasetDir)
    
#%% Loading each files
startTime = time.time()

wholeData = {}
metadata = {}
dataset_names = []

for path in args.datasetPath:
    with h5py.File(path, 'r') as inData:
        
        print(f"Loading {path.split('/')[-1]} ...")
        
        ## Getting available image sizes
        availImageSizes = list(inData.keys())
        
        for imageSize in availImageSizes :
            
            print(f"\tLoading IS: {imageSize}")
            
            ## Getting available bin sizes for the given image size 
            availBinSizes = list(inData[imageSize].keys())
            
            if imageSize not in wholeData.keys():
                wholeData[imageSize] = {}
            
            for binSize in availBinSizes :
                
                print(f"\t\tLoading BS: {binSize}")
                
                ## Adding the binsize key to wholeData if not present
                if binSize not in wholeData[imageSize].keys() :
                    wholeData[imageSize][binSize] = {
                        "x" : np.array(inData[imageSize][binSize]["x"]),
                        "y" : np.array(inData[imageSize][binSize]["y"]),
                        "names" : np.array(inData[imageSize][binSize]["names"]),
                        "groups" : np.array(inData[imageSize][binSize]["groups"])
                        }
                
                ## Else concatenating new data to the already loaded data
                else : 
                    wholeData[imageSize][binSize]["x"] = np.concatenate(
                        (wholeData[imageSize][binSize]["x"], 
                         np.array(inData[imageSize][binSize]["x"])),
                        axis = 0)
                    
                    wholeData[imageSize][binSize]["y"] = np.concatenate(
                        (wholeData[imageSize][binSize]["y"], 
                         np.array(inData[imageSize][binSize]["y"])),
                        axis = 0)
                    
                    wholeData[imageSize][binSize]["names"] = np.concatenate(
                        (wholeData[imageSize][binSize]["names"], 
                         np.array(inData[imageSize][binSize]["names"])),
                        axis = 0)
                    
                    wholeData[imageSize][binSize]["groups"] = np.concatenate(
                        (wholeData[imageSize][binSize]["groups"], 
                         np.array(inData[imageSize][binSize]["groups"])+np.max(wholeData[imageSize][binSize]["groups"])+1),
                        axis = 0)
        
        # Reading metadata
        dataset_name = inData.attrs["dataset"]
        for key, attributes in inData.attrs.items():
            metadata[f"{dataset_name}|{key}"] = attributes

print(f"Done ! ({round(time.time()-startTime,3)}s)")

#%% Writing to H5

startTime = time.time()
           
with h5py.File(args.outputDir, "w") as out:
    print(f"Writing data to {args.outputDir}")
    
    for imgSize in wholeData.keys():
        print(f"\tCreating {imgSize} group")
        imgSizeGroup = out.create_group(str(imgSize))
        
        for binSize in wholeData[imgSize].keys():
            print(f"\t\tAdding BS: {binSize} to IS: {imgSize}")
            binSizeGroup = imgSizeGroup.create_group(str(binSize))
            
            for key in wholeData[imgSize][binSize].keys():
                print(f"\t\t\tAdding {key} to {imgSize}:{binSize}")

                binSizeGroup.create_dataset(
                    key,
                    data=wholeData[imgSize][binSize][key],
                    compression = "gzip"
                    )
    
    # Adding datasets metadata
    for key, value in metadata.items():
        out.attrs[key] = value

    # Adding Multi2One metadata
    out.attrs["dataset"] = args.dataset_name
    out.attrs["Multi2One.version"] = version
    out.attrs["Multi2One.creationDate"] = datetime.datetime.now().strftime("%m/%d/%Y-%H:%M:%S") 

print(f"Done ! ({round(time.time()-startTime,3)}s)")
