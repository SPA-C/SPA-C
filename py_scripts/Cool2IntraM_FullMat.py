#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cool to Intra-Contig matrices
Create the dataset for chimeric contig detection.

@author: Alexis Mergez
@Last modified: 2025/12/05
@version: 1.8.1
"""

import os
import time
import datetime
import cooler
import numpy as np
import h5py
import argparse
import json
from tqdm import tqdm

version = "1.8.1"

#%% Functions
def parse_breakpoints(
        breakpoint_file: str,
        bin_size: int
    ):
    """
    Parse the given breakpoint file (JSON) and return a dictionary: {contig_name: list of breakpoint(s) bin}

    Inputs:
        - breakpoint_file: str. Path to the breakpoint file.
        - bin_size: int. Genomic range of a bin in the HiC matrix

    Output:
        - dict. Keys are contig name, values are list of breakpoint(s) bin
    """
    # Reading JSON
    with open(breakpoint_file, "r") as handle:
        breakpoint_gen = json.load(handle)

    # Converting genomic position into bin position
    breakpoint_bins = {}
    for key, bin_positions in breakpoint_gen.items():
        breakpoint_bins[key] = [k // bin_size for k in bin_positions]

    return breakpoint_bins

def get_sub_array(
        whole_array: np.array,
        start_bin: int,
        image_size: int,
        threshold: int=None
    ):
    """
    Returns forward and reverse matrices (2, image_size*2, image_size*2).
    The reverse is simply the forward rotated by 180°.

    Inputs:
        - whole_array: np.array. Contig HiC matrix returned when fetched by cooler.
        - start_bin: int. Center bin number of the image.
        - image_size: int. Image size of the dataset x arrays.
        - threshold: int. Filter on minimal counts in matrix (default = None).

    Output:
        - dict. Dict with keys being the corner and values being the matrix of shape : (1, image_size*2, image_size*2).
    """
    arrays = {}

    # Forward
    f_arr = np.expand_dims(
            whole_array[
                (start_bin - image_size):(start_bin + image_size),  # Y
                (start_bin - image_size):(start_bin + image_size)  # X
            ],
        0)

    if threshold is None or np.sum(f_arr) >= threshold:
        assert f_arr.shape == (1, 2*image_size, 2*image_size)
        arrays["C2"] = f_arr

    # Reverse
    r_arr = np.expand_dims(
            np.rot90(
                f_arr[0,:,:], 2),
        0)
    if threshold is None or np.sum(r_arr) >= threshold:
        assert r_arr.shape == (1, 2*image_size, 2*image_size)
        arrays["C3"] = r_arr

    return arrays

def import_intra_matrices(
        image_size: int,
        cool: cooler,
        contig: str,
        contig_size: int,
        threshold: int=None,
        breakpoints: dict=None,
        only_negatives: bool=False,
        name_dict: dict=None,
        padding: int=None
    ):
    """
    Import intra-contig matrices for the 2 given contigs.

    Inputs:
        - contig : str. Contig1 name.
        - cool : cool_object. Whole HiC matrix opened wih cooler.
        - image_size : int. Width of the image (same as height).
        - threshold: int. Filter on minimal counts in matrix (default = None).
        - breakpoints: dict. Output of parse_breakpoints() (Default is None).
        - only_negatives: bool. Set if only negative samples are needed (default = False).
        - name_dict: dict. Name conversion dictionary.

    Outputs:
        - x : numpy array (n, image_size, image_size). All 4 HiC matrices.
        - names : numpy array (n, 1). Matrices names (ctg1:pos1|ctg2:pos2).
                  pos1 and pos2 equals 0 at the beginning of ctg1, or 1 at the end of ctg1.
    """
    # Initializing storing list
    x, names, groups, y = [], [], [], []

    # Initializing UID variables
    groupUID = 0

    # Fetching the array
    whole_array = cool.matrix(balance=False).fetch(
        (contig, None, None),
        (contig, None, None)
    )

    if padding is not None and whole_array.shape[0] < 2*image_size:
        _ = np.ones((2*image_size, 2*image_size)) * padding
        _[:whole_array.shape[0],:whole_array.shape[0]] = whole_array
        whole_array = _

    # Getting the list of centered starting bin
    positions = range(
        image_size,
        contig_size - image_size + 1
    )

    for pos in positions: # Left most bin number list
        ## Fetching sub-matrices from the whole hic matrix
        try:
            arrays = get_sub_array(
                whole_array = whole_array,
                start_bin = pos,
                image_size = image_size,
                threshold = threshold
            )

        except Exception as e:
            arrays = {}
            print(f"Unable to parse {contig}:{pos}\n{e}")

        if len(arrays): # arrays is not empty

            if breakpoints is not None: # Getting the label for the imported arrays
                label = get_y(
                    breakpoints=breakpoints,
                    contig_name=contig,
                    position=pos,
                    image_size=image_size
                )
            else: label = 1 # No breakpoints given means no breakpoint in the dataset

            if label is None:
                print(f"{contig}_{pos}")
                raise ValueError

            if (only_negatives and label == 0) or (not only_negatives):
                for corner, arr in arrays.items():
                    # Adding the matrix to the storing list
                    x.append(arr)

                    # Adding the name

                    if name_dict is not None:
                        name = name_dict[contig]
                    else:
                        name = contig

                    names.append(
                        f"{name}_{pos}_{corner}"
                    )

                    y.append(label)

                    groups.append(groupUID)

    if len(x): # If even one matrix has been imported
        return x, y, names, groups

    else: # If no matrix has been imported
        return None, None, None, None

def get_whole_matrices(
        cool: cooler.Cooler,
        chrom_names: list
    ):
    """
    Read the whole matrices from the COOL file.

    Inputs:
    - cool: cooler.Cooler. HiC matrix (MCOOL) imported using cooler api.
    - chrom_names: list. List of chromosomes present in the cool file.

    Output:
    - dict. Keys are contig names, values are whole HiC matrix.
    """
    whole_matrices = {
        contig: np.array(cool.matrix(balance=False).fetch((contig, None, None), (contig, None, None)))
        for contig in chrom_names
    }

    return whole_matrices

def get_y(
        breakpoints: dict,
        contig_name: str,
        position: int,
        image_size: int,
        tolerance: int=7
    ):
    """
    Return Y based on breakpoints. If the breakpoint (+- tolerance) is visible in the matrix, returns 0; otherwise, returns 1.

    Inputs:
        - breakpoints: dict. Output of parse_breakpoints()
        - contig_name: str. Name of the contig.
        - position: int. Bin number.
        - image_size: int. Number of bin in x array
        - tolerance: int. Matrices whose center bin is closer than (image_size-tolerance) bin is labelled 0.

    Output:
        - Y label for the given position.
    """

    if contig_name in breakpoints.keys():
        # Checking each breakpoint bin (0 if not in the range, else 1)
        check = [
            bkp_bin-image_size+tolerance <= position <= bkp_bin+image_size-tolerance
            for bkp_bin in breakpoints[contig_name]
        ]

        return int(np.sum(check) == 0) # The sum equals 0 if not in range, hence label should be 1

    else :
        return 1

def create_intra_contig_dataset(
        bin_size: int,
        image_size: int,
        cool: cooler.Cooler,
        threshold: int=None,
        breakpoints: dict=None,
        only_negatives: bool=False,
        name_dict: dict=None,
        padding: int=None
    ):
    """
    Create the dataset of intra-contig matrices.

    Inputs:
        - bin_size: int. Genomic range of the matrix bin (or pixel).
        - image_size: int. Size of the HiC matrix.
        - cool: cooler.Cooler. HiC matrix (MCOOL) imported using cooler api.
        - threshold: int. Filter on minimal counts in matrix (default = None).
        - breakpoints: dict. Output of parse_breakpoints() (Default is None).
        - only_negatives: bool. Set if only negative samples are needed (default = False).
        - name_dict: dict. Name conversion dictionary.

    Output:
        - dict. Data dict with x, y, names and groups keys.
    """

    data = {}
    all_x, all_names, all_groups, all_y = [], [], [], []
    group_max = 0

    chrom_sizes = cool.chromsizes.to_dict()
    contigs = list(chrom_sizes.keys())

    for contig in tqdm(contigs, desc=f"Importing intra-contigs ...", unit="ctg"):

        # Fetching matrices
        x, y, names, groups = import_intra_matrices(
            contig=contig,
            contig_size=(chrom_sizes[contig]//bin_size),
            cool=cool,
            image_size=image_size,
            threshold=threshold,
            breakpoints=breakpoints,
            only_negatives=only_negatives,
            name_dict=name_dict,
            padding=padding
        )

        # Appending all_x if not empty
        if x is not None:
            all_x += x
            all_names += names

            all_groups += [k+group_max+1 for k in groups]
            group_max = np.max(all_groups)

            all_y += y

    # Adding all_x, all_names and all_groups to data dict
    if len(all_x) >= 1:
        data[(image_size, bin_size)] = {
            "x": np.concatenate(all_x, axis=0),
            "y": np.array(all_y),
            "groups": np.array(all_groups),
            "names": np.array(all_names)
        }

    return data

def export_to_HDF5(
        data: dict,
        whole: dict,
        output: str,
        metadata: dict
    ):
    """
    Exporting the dataset to HDF5.

    Inputs:
        - data: dict. Dataset created using import_inter_matrices().
        - whole: dict. Dictionary created with get_whole_matrices().
        - output: str. Path to the HDF5 output file.
        - metadata: dict. Metadata dictionary to add to the HDF5.
    """
    image_size, bin_size = list(data.keys())[0]

    with h5py.File(output, "w") as out:

        print(f"Writing data to {output}")
        print(f"\tCreating {image_size} group")
        image_size_group = out.create_group(str(image_size))

        print(f"\t\tAdding {bin_size} to {image_size}")
        bin_size_group = image_size_group.create_group(str(bin_size))

        for key in data[(image_size, bin_size)].keys():

            print(f"\t\tAdding {key} to {image_size}:{bin_size}")

            if key != "names":
                bin_size_group.create_dataset(
                    key,
                    data=data[(image_size, bin_size)][key],
                    compression = "gzip"
                    )
            else :
                bin_size_group.create_dataset(
                    key,
                    data=data[(image_size, bin_size)][key].astype('S'),
                    compression = "gzip"
                    )

        print(f"\t\tAdding whole matrices to {image_size}:{bin_size}")
        whole_group = bin_size_group.create_group("whole")
        for key, arr in whole.items():
            whole_group.create_dataset(
                key,
                data=arr,
                compression="gzip"
            )

        for key, value in metadata.items():
            out.attrs[key] = value

def main(
        cool_file: str,
        image_size: int,
        bin_size: int,
        output: str,
        threshold: int=None,
        breakpoint_file: str=None,
        only_negatives: bool=False,
        name_table: str=None,
        padding: int=None
    ):
    """
    Main function to create the dataset, normalize it and output it to HDF5

    Inputs:
        - cool_file: str. Path to the MCOOL.
        - image_size: int. Size of the image to import.
        - bin_size: int. Genomic range of HiC bin.
        - output: str. Path to the output HDF5.
        - threshold: int. Filter on minimal counts in matrix (default = None).
        - breakpoint_file: str. Path to the breakpoint file (default = None).
        - only_negatives: bool. Set if only negative samples are needed (default = False).
        - name_dict: dict. Name conversion dictionary.
    """
    print("Converting cool file to DLScaff scaffolding dataset")

    # Getting other variables from inputs
    dataset_name = os.path.splitext(os.path.basename(cool_file))[0]

    # Reading COOL file
    try:
        cool = cooler.Cooler(cool_file + f"::/resolutions/{bin_size}")
    except Exception as E1:
        print(
            f"Unable to open {cool_file}::/resolutions/{bin_size}:\n{E1}\nAttempting to load without specific resolution.")
        try:
            cool = cooler.Cooler(cool_file)
        except Exception as E2:
            print(f"Unable to load :\n{E2}")

    chrom_sizes = cool.chromsizes.to_dict()

    # Parsing breakpoints
    if breakpoint_file is not None:
        breakpoints = parse_breakpoints(
            breakpoint_file=breakpoint_file,
            bin_size=bin_size
        )
    else: breakpoints = None

    # Renaming contigs if necessary
    if name_table is not None:
        with open(name_table, "r") as handle:
            name_dict = {k.strip().split('\t')[0]: k.strip().split('\t')[1] for k in handle.readlines()}
    else: name_dict = None

    # Creating the dataset
    data = create_intra_contig_dataset(
        bin_size=bin_size,
        image_size=image_size,
        cool=cool,
        threshold=threshold,
        breakpoints=breakpoints,
        only_negatives=only_negatives,
        name_dict=name_dict,
        padding=padding
    )

    # Fetching whole matrices
    whole_matrices = get_whole_matrices(
        cool=cool,
        chrom_names=list(chrom_sizes.keys())
    )

    # Sanity checks
    for key in data.keys():
        assert data[key]["x"].shape[0] == data[key]["y"].shape[0]
        assert data[key]["x"].shape[0] == data[key]["groups"].shape[0]
        assert data[key]["x"].shape[0] == data[key]["names"].shape[0]

    # Exporting to HDF5
    metadata = {
        "Cool2IntraM_FullMat.version": version,
        "data.type": "Intra.FullMat",
        "dataset": dataset_name,
        "Cool2IntraM_FullMat.filter": threshold if threshold is not None else "None",
        "Cool2IntraM_FullMat.creation_date": datetime.datetime.now().strftime("%m/%d/%Y-%H:%M:%S")
    }

    export_to_HDF5(
        data=data,
        whole=whole_matrices,
        output=output,
        metadata=metadata
    )

    print("Done !")

if __name__ == '__main__':
    #%% Parsing arguments
    argParser = argparse.ArgumentParser(
        description= f"Scaffolding dataset - v{version}"
        )
    argParser.add_argument(
        "--cool",
        "-c",
        dest = "cool_file",
        required = True,
        help = "Directory of cool file",
        type = str
        )
    argParser.add_argument(
        "--image-size",
        "-i",
        type = int,
        dest = "image_size",
        default = 10,
        help = "Sizes of images (Default: 10)"
        )
    argParser.add_argument(
        "--bin-size",
        "-b",
        type = int,
        dest = "bin_size",
        default = 5000,
        help = "Genomic size of a bin (Default: 5000)"
        )
    argParser.add_argument(
        "--threshold",
        dest = "threshold",
        required = False,
        help = "Minimum counts in matrix",
        type = int,
        default = None
        )
    argParser.add_argument(
        "--output",
        dest = "output",
        required = True,
        help = "Output dataset (HDF5)",
        type = str
        )
    argParser.add_argument(
        "--breakpoint",
        dest="breakpoint_file",
        required=False,
        help="Breakpoint file (JSON)",
        type=str,
        default=None
        )
    argParser.add_argument(
        "--only-negatives",
        dest="only_negatives",
        action='store_true',
        help="Keep only the negative matrices"
    )
    argParser.add_argument(
        "--name-table",
        dest="name_table",
        type=str,
        required=False,
        help="Renaming table for contig names",
        default=None
    )
    argParser.add_argument(
        "--padding",
        dest="padding",
        type=int,
        required=False,
        help="Add padding to very small contigs",
        default=None
    )
    args = argParser.parse_args()

    main(
        cool_file = args.cool_file,
        image_size = args.image_size,
        bin_size = args.bin_size,
        threshold = args.threshold,
        output = args.output,
        breakpoint_file = args.breakpoint_file,
        only_negatives = args.only_negatives,
        name_table=args.name_table,
        padding=args.padding
    )