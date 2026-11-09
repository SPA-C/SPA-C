#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils functions for DLScaff

@author: Alexis Mergez
@Last modified: 2025/01/16
@version: 4.4
"""

import pandas as pd
import numpy as np
import os
import json
import h5py


class Interval():
    """
    Simple class for working with alignments
    """
    
    def __init__(self, start, end):
        self.start = start
        self.end = end
        
    def isDisjoined(self, interval):
        """
        Check if self is disjoined from provided interval.

        Parameters
        ----------
        interval : Interval
            Interval to compare with.

        Returns
        -------
        bool
            Check results.

        """
        return (self.end < interval.start or self.start > interval.end)
    
    def isOverlaped(self, interval):
        return not self.isDisjoined(interval)
    
    def getOverlap(self, interval):
        if self.isDisjoined(interval) : return False
        return Interval(
            start = max(self.start, interval.start),
            end = min(self.end, interval.end)
            )
    
    def isIncluded(self, interval):
        _ = self.getOverlap(interval)
        if _ is False:
            return False
        return (_.start == self.start and _.end == self.end)
    
    def isIncludedOnce(self, intervalList):
        for interval in intervalList:
            if self.isIncluded(interval):
                return True
        return False
    
    def isSame(self, interval):
        return (self.start == interval.start and self.end == interval.end)
    
    def getLimits(self):
        return [self.start, self.end]
            

def load_paf(path, min_contig_length = 0, debug = True):
    
    ## Reading csv
    paf = pd.read_csv(path, delimiter="\t", header = None, usecols = [i for i in range(12)])
    paf.columns = ["Q_name", "Q_len", "Q_start", "Q_end", "Strand", "T_name", "T_len", 
                   "T_start", "T_end", "N_residue", "Al_len", "Quality"]

    ## Filtering contigs
    n_contigs = len(paf.Q_name.unique())
    paf = paf[paf["Q_len"] > min_contig_length]

    if debug : print(f"\033[43m{len(paf.Q_name.unique())} contigs remaining (out of {n_contigs})\033[0m")

    ## Adding proportion of alignment according to contig size
    paf.loc[:,"Al_len"] = paf["Q_end"]-paf["Q_start"]
    paf.loc[:,"Al_len_log"] = np.log(paf["Al_len"])
    paf.loc[:,"Al_proportion"] = paf["Al_len"]/paf["Q_len"]
    
    return paf

def load_json(path):
    """
    Parsing a json into a dictionnary.

    Parameters
    ----------
    path : str
        Path to the json file.

    Returns
    -------
    data : dict
        Dictonnary containing data of the json file.

    """
    if not os.path.exists(path):
        raise ImportError(f"{path} not found")
    
    ## Reading file
    with open(path, "r") as file:
        data = json.load(file)
        
    return data

def save_to_json(data, path):
    ## Checking if data is a dictionnary
    if not isinstance(data, dict):
        raise TypeError("data is not a dictionnary !")
    
    ## Getting the folder path
    folder_path = path[:len(path)-len(path.split("/")[-1])-1]
    
    ## Checking if folder path exists
    if not os.path.isdir(folder_path) :
        os.makedirs(folder_path)
        
    ## Exporting the dictionnary to json
    with open(path, "w") as out:
        json.dump(data, out, indent=4)

def load_breakpoints(path): 
    
    if not os.path.exists(path):
        raise ImportError(f"{path} not found")

    breakpoints = load_json(path)
    return breakpoints  

def check_dir(path, create = False):
    """
    Check if the directory exists and create it if asked to. 

    Parameters
    ----------
    path : str
        Path to the directory to check.
    create : bool, optional
        Control if the directory have to be created if needed. 
        The default is False.

    Raises
    ------
    FileNotFoundError
        Path does not exist and create is set to False.

    Returns
    -------
    None.

    """
    ## Checking if path exists
    if not os.path.isdir(path):
        if create:
            print(f"{path} does not exists, creating it ...")
            os.makedirs(path)
        else:
            raise FileNotFoundError(f"No such directory: {path}")
            
def check_path(path):
    """
    Check if the path exists.

    Parameters
    ----------
    path : str
        Path to check.

    Raises
    ------
    FileNotFoundError
        Path do not exists.

    Returns
    -------
    None.

    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file or directory {path}")
            
def to_h5(data, path, create = True):
    """
    Export a dictionnary to a hdf5.
    Made for creating big datasets.

    Parameters
    ----------
    data : dict
        Dictionnary.
    path : str
        Path to the output file (including filename).
    create : bool, optional
        Control if the directory have to be created if needed. 
        The default is True.

    Raises
    ------
    TypeError
        Data is not a dictionnary.

    Returns
    -------
    None.

    """
    
    ## Checking if data is a dictionnary
    if not isinstance(data, dict):
        raise TypeError("data is not a dictionnary !")
    
    ## Getting the folder path
    folder = path[:len(path)-len(path.split("/")[-1])-1]
    
    ## Checking if folder path exists
    check_dir(folder, create)
    
    ## Exporting the dictionnary to h5
    with h5py.File(path, "w") as out:
        for key in data.keys():
            print(f"Adding {key} to {path} ...")
            if isinstance(data[key], dict):
                grp = out.create_group(key)
                for sub_key in data[key].keys():
                    grp.create_dataset(sub_key, data=data[key][sub_key], compression="lzf")
            else:
                out.create_dataset(key, data=data[key], compression="lzf")
    
    ## Inline checkup
    print(f"Dataset saved as: {path}")
    
def load_h5(path):
    """
    Parsing a hdf5 into a dictionnary.

    Parameters
    ----------
    path : str
        Path to the hdf5 file.

    Returns
    -------
    data : dict
        Dictonnary containing data of the hdf5 file.

    """
    ## Checking path
    check_path(path)
    
    file = h5py.File(path, 'r')
    
    data = {}
    for key in file.keys():
        
        try : # If it's a dataset
            values = file[key]
            if values.dtype == "O":
                data[key] = [k.decode("ASCII") for k in values]
            
            else:
                data[key] = np.array(values)
        
        except : #If we have a subgroup
            data[key] = {}
            for sub_key in file[key].keys():
                values = file[key][sub_key]
                if values.dtype == "O":
                    data[key][sub_key] = [k.decode("ASCII") for k in values]
                
                else:
                    data[key][sub_key] = np.array(values)
        
            
    return data


