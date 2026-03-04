#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cool to Inter-Contig matrices
Create the dataset for scaffolding using corrected contigs.

@author: Alexis Mergez
@Last modified: 2025/12/17
@version: 1.11
"""

import os
import time
import re
import datetime
import cooler
import numpy as np
import scipy.sparse as sparse
import h5py
import argparse
import string
import concurrent.futures
import cProfile
from tqdm import tqdm
from numpy.lib.stride_tricks import as_strided

version = "1.11"

#%% Functions
def parse_corrected_contigs(
        contig_file: str,
        bin_size: int,
        image_size: int,
        chrom_sizes: dict,
        size_offset = 6
    ):
    """
    Read the corrected contigs file (TXT) produced by Chimera_predictor.py and outputs contig names and ranges.

    Input:
    - contig_file: str. Path to the contig file.
    - bin_size: int. Size of a bin.
    - image_size: int. Width of the image

    Output:
    - contigs: list. List of corrected contig names.
    """
    all_contigs, contigs, base_contigs, seen_contigs = [], [], {}, set()

    with open(contig_file, "r") as handle:
        for line in handle:
            all_contigs.append(line.rstrip())

    for full_name in all_contigs:
        ctg, start, end = re.split(r":|-", full_name)
        start = int(start)//bin_size
        end = int(end)//bin_size
        seen_contigs.add(ctg)

        if (int(end)-int(start)) < image_size-size_offset: # Filtering small contigs. (Minimum 6 bins)
            print(f"Unable to use {full_name}: length {(int(end)-int(start))*bin_size} < minimum contig size !")
        else:
            contigs.append(f"{ctg}:{start}-{end}")
            base_contigs[f"{ctg}:{start}-{end}"] = full_name

    # Adding contigs that could be used in scaffolding but were not screened during chimera detection
    for chrom_name, length in chrom_sizes.items():
        if chrom_name not in seen_contigs:
            if (length // bin_size) < image_size-size_offset:  # Checking if the contig is actually long enough
                print(f"Unable to use {chrom_name}: length {length} < minimum contig size !")
                continue

            contigs.append(f"{chrom_name}:0-{(length // bin_size)}")

    return contigs, base_contigs

def fetching_inter_quadrant(ctg1, pos1, end1, size1, ctg2, pos2, end2, size2, image_size, mat, offsets):
    start_pos1 = (pos1-min(size1, image_size)) if pos1==end1 else pos1
    end_pos1 = (pos1+min(size1, image_size)) if pos1!=end1 else pos1
    start_pos2 = (pos2 - min(size2, image_size)) if pos2 == end2 else pos2
    end_pos2 = (pos2 + min(size2, image_size)) if pos2 != end2 else pos2

    # OLD ---
    # arr = mat.fetch( # Fetching matrix using genomic coordinates
    #                 (ctg1, start_pos1 * bin_size, end_pos1 * bin_size),
    #                 (ctg2, start_pos2 * bin_size, end_pos2 * bin_size)
    #             )

    i1_0 = offsets[ctg1] + start_pos1
    i1_1 = offsets[ctg1] + end_pos1
    i2_0 = offsets[ctg2] + start_pos2
    i2_1 = offsets[ctg2] + end_pos2

    return mat[i1_0:i1_1, i2_0:i2_1].toarray().astype(np.int32)

def fetching_intra_quadrant(ctg, pos, end, ctg_size, image_size, mat, offsets):
    start_pos = (pos - min(ctg_size, image_size)) if pos == end else pos
    end_pos = (pos + min(ctg_size, image_size)) if pos != end else pos

    # OLD ---
    # arr = mat.fetch(  # Fetching matrix using genomic coordinates
    #     (ctg1, start_pos * bin_size, end_pos * bin_size),
    #     (ctg1, start_pos * bin_size, end_pos * bin_size)
    # )

    i0 = offsets[ctg] + start_pos
    i1 = offsets[ctg] + end_pos

    return mat[i0:i1, i0:i1].toarray().astype(np.int32)

def parse_intra_quadrants(contigs, image_size, mat, offsets):
    intra_quadrants = {}
    for contig in tqdm(contigs, desc="Parsing intra quadrants", unit="ctg", leave=True):
        ctg, start, end = re.split(r'-|:', contig)
        start, end = int(start), int(end)

        for pos in [start, end]:
            try :
                intra_quadrants[contig][pos] = fetching_intra_quadrant(ctg, pos, end, end-start, image_size, mat, offsets)
            except :
                intra_quadrants[contig] = {
                    pos: fetching_intra_quadrant(ctg, pos, end, end-start, image_size, mat, offsets)
                }

    return intra_quadrants

def import_inter_matrices(
        image_size: int,
        bin_size: int,
        intra_quadrants: dict,
        mat: cooler.Cooler.matrix,
        contig1: str,
        contig2: str,
        chrom_sizes: dict,
        offsets: dict,
        base_contigs: dict,
        threshold: int=None,
        padding: int=-1
    ):
    """
    Import inter-contig matrices for the 2 given contigs.
    Since the cooler API only give the adjacency matrix between 2 contigs, the whole matrix is reconstructed manually.

    Inputs:
        - contig1 : str. Contig1 name.
        - contig2 : str. Contig2 name.
        - cool : cool_object. Whole HiC matrix opened wih cooler.
        - image_size : int. Width of the image (same as height).
        - bin_size : int. Pixel resolution of the HiC matrix.
        - threshold: int. Filter on minimal counts in matrix (default = None).

    Outputs:
        - x : numpy array (n, image_size, image_size). Both forward and reverse matrices.
        - names : numpy array (n, 1). Matrices names (ctg1:pos1|ctg2:pos2).
                  pos1 and pos2 equals 0 at the beginning of ctg1, or 1 at the end of ctg1.
    """
    # Getting contig names, starts and ends. (ctg_name:start_bin-end_bin)
    ctg1, start1, end1 = re.split(r'-|:', contig1)
    ctg2, start2, end2 = re.split(r'-|:', contig2)

    # Converting to int
    end1, end2 = int(end1), int(end2)
    start1, start2 = int(start1), int(start2)
    size1, size2 = end1-start1, end2-start2

    # Initializing lists
    x, names = [], []

    for pos1 in [start1, end1]:  # Start or end on ctg1
        for pos2 in [start2, end2]:  # Start or end on ctg2
            ## Fetching adjacency matrix using cooler API
            assert (start1 <= end1 and start2 <= end2)

            # Correcting array orientation depending on contig ends used
            if (pos1 == start1 and pos2 == end2):
                arr = fetching_inter_quadrant(
                    ctg1=ctg2, pos1=pos2, end1=end2, size1=size2, ctg2=ctg1, pos2=pos1, end2=end1, size2=size1,
                    image_size=image_size, mat=mat, offsets=offsets
                )
            else:
                tmp_arr = fetching_inter_quadrant(
                    ctg1=ctg1, pos1=pos1, end1=end1, size1=size1, ctg2=ctg2, pos2=pos2, end2=end2, size2=size2,
                    image_size=image_size, mat=mat, offsets=offsets
                )
                if (pos1 == start1 and pos2 == start2):
                    arr = np.flipud(tmp_arr)

                elif (pos1 == end1 and pos2 == end2):
                    arr = np.fliplr(tmp_arr)

                else:
                    arr = tmp_arr

            try:
                if (threshold is None or np.sum(arr) >= threshold):
                    # Reconstructing the full matrix
                    # We need to make sure to rotate diagonals quadrant if needed (if top left is start and if bottom right is end) !
                    fm = np.ones((image_size * 2, image_size * 2), dtype=np.int32) * padding

                    ## Adding adjacency matrix
                    status = None
                    if pos1 == start1 and pos2 == end2: # S1E2
                        fm[max((image_size - size2), 0):image_size, image_size:min((image_size + size1), 2*image_size)] = arr
                        fm[image_size:min((image_size + size1), 2*image_size), max((image_size - size2), 0):image_size] = np.rot90(np.flip(arr, axis=1), k=1)
                    else :
                        status = "01"
                        fm[max((image_size - size1), 0):image_size, image_size:min((image_size + size2), 2*image_size)] = arr
                        status = "02"
                        fm[image_size:min((image_size + size2), 2*image_size), max((image_size - size1), 0):image_size] = np.rot90(np.flip(arr, axis=1), k=1)

                    ## Adding diagonals
                    if pos1 == start1 and pos2 != end2: # S1S2
                        status = "A"
                        fm[max((image_size - size1), 0):image_size, max((image_size - size1), 0):image_size] = np.rot90(intra_quadrants[contig1][pos1], 2)
                        fm[image_size:min((image_size + size2), 2*image_size), image_size:min((image_size + size2), 2*image_size)] = intra_quadrants[contig2][pos2]
                    elif pos1 == start1 and pos2 == end2: # S1E2
                        status = "B"
                        fm[max((image_size - size2), 0):image_size, max((image_size - size2), 0):image_size] = intra_quadrants[contig2][pos2]
                        fm[image_size:(image_size+size1), image_size:(image_size+size1)] = intra_quadrants[contig1][pos1]
                        fm = np.rot90(fm, 2)
                    elif pos1 != start1 and pos2 == end2: # E1E2
                        status = "C"
                        fm[max((image_size - size1), 0):image_size, max((image_size - size1), 0):image_size] = intra_quadrants[contig1][pos1]
                        fm[image_size:min((image_size + size2), 2*image_size), image_size:min((image_size + size2), 2*image_size)] = np.rot90(intra_quadrants[contig2][pos2], 2)
                    else: # E1S2
                        status = "D1"
                        fm[max((image_size - size1), 0):image_size, max((image_size - size1), 0):image_size] = intra_quadrants[contig1][pos1]
                        status = "D2"
                        fm[image_size:min((image_size + size2), 2*image_size), image_size:min((image_size + size2), 2*image_size)] = intra_quadrants[contig2][pos2]

                    # Adding the matrix to the storing list
                    x.append(fm)

                    # Adding the reverse matrix
                    x.append(
                        np.rot90(np.flip(fm, axis=1), -1)
                    )

                    # Adding the name (Compensating end shift)
                    try :
                        full_ctg1 = base_contigs[f"{ctg1}:{start1}-{end1}"]
                    except :
                        full_ctg1 = f"{ctg1}:{start1*bin_size}-{end1*bin_size if (chrom_sizes[ctg1]//bin_size) != end1 else chrom_sizes[ctg1]}"

                    try :
                        full_ctg2 = base_contigs[f"{ctg2}:{start2}-{end2}"]
                    except :
                        full_ctg2 = f"{ctg2}:{start2*bin_size}-{end2*bin_size if (chrom_sizes[ctg2]//bin_size) != end2 else chrom_sizes[ctg2]}"

                    names.append(
                        # Position equals 0 at the start of the contig, 1 at the end.
                        f"{full_ctg1}:{int(pos1 != start1)}|{full_ctg2}:{int(pos2 != start2)}"
                    )
                    names.append(
                        f"{full_ctg2}:{int(pos2 != start2)}|{full_ctg1}:{int(pos1 != start1)}"
                    )

                else:
                    full_ctg1 = f"{ctg1}:{start1}-{end1}"
                    full_ctg2 = f"{ctg2}:{start2}-{end2}"
                    print("Unable to add :", f"{full_ctg1}:{int(pos1 != start1)}|{full_ctg2}:{int(pos2 != start2)}",
                          "Incorrect shape")

            except Exception as inst:
                # Skipping if unable to retrieve the matrix (may be due to too short contigs)
                print("Unable to add :", f"{contig1}:{int(pos1 != start1)}|{contig2}:{int(pos2 != start2)}, status: {status}, {arr.shape}",
                      "Exception")
                print(inst)
                pass

    if len(x): # If even one matrix has been imported
        return x, names

    else: # If no matrix has been imported
        return None, None

def strideConv(
        image: np.array,
        weight: np.array,
        stride: int
    ):
    """
    Strided convolution.

    """
    im_h, im_w = image.shape
    f_h, f_w = weight.shape

    out_shape = (1 + (im_h - f_h) // stride, 1 + (im_w - f_w) // stride, f_h, f_w)
    out_strides = (image.strides[0] * stride, image.strides[1] * stride, image.strides[0], image.strides[1])
    windows = as_strided(image, shape=out_shape, strides=out_strides)

    return np.tensordot(windows, weight, axes=((2, 3), (0, 1)))

def import_inter_matrices_hotspot(
        image_size: int,
        bin_size: int,
        mat: cooler.Cooler.matrix,
        ctg1: str,
        ctg2: str,
        threshold: int = None,
        steps: int = 1
    ):
    """
    Import inter-contig matrices for the 2 given contigs,
    but search for counts hotspots within the whole adjacency matrix.

    Inputs:
        - ctg1 : str. Contig1 name.
        - ctg2 : str. Contig2 name.
        - cool : cool_object. Whole HiC matrix opened wih cooler.
        - image_size : int. Width of the image (same as height).
        - bin_size : int. Pixel resolution of the HiC matrix.
        - threshold: int. Filter on minimal counts in matrix (default = None).
        - step: int. Step when importing matrices.

    Outputs:
        - x : numpy array (n, image_size, image_size). Both forward and reverse matrices.
        - names : numpy array (n, 1). Matrices names (ctg1:pos1:side|ctg2:pos2:side).
                  pos1 and pos2 equals 0 at the beginning of ctg1, or 1 at the end of ctg1.
    """
    # Searching best subpatch to import
    stride = 1000

    whole_mat = mat.fetch(  # Fetching whole adjencency matrix
        (ctg1, None, None),
        (ctg2, None, None)
    )

    # First search with big patches
    big_patches = strideConv(whole_mat.astype(np.int32), np.ones((stride, stride)), stride)
    bigP_idx = np.where(big_patches >= np.max(big_patches) / 2)

    selected_patches = []
    for x, y in zip(bigP_idx[0], bigP_idx[1]):
        patch_arr = whole_mat[x * (stride):x * (stride) + stride, y * (stride):y * (stride) + stride]

        # Searching best sub_patch
        sub_patch_size, sub_patch_stride = stride // 10, stride // 8
        sub_patches = strideConv(patch_arr, np.ones((sub_patch_size, sub_patch_size)), sub_patch_stride)
        subpatch_idx = np.where(sub_patches == np.max(sub_patches))

        selected_patches.append((  # (X_start, X_end, Y_start, Y_end)
            x * (stride) + subpatch_idx[0][0] * (sub_patch_stride),
            x * (stride) + subpatch_idx[0][0] * (sub_patch_stride) + (sub_patch_size),
            y * (stride) + subpatch_idx[1][0] * (sub_patch_stride),
            y * (stride) + subpatch_idx[1][0] * (sub_patch_stride) + (sub_patch_size)
        ))

    # Initializing lists
    arrays, names = [], []
    diags = {}
    
    for xstart, xend, ystart, yend in selected_patches:
        for x in range(xstart, xend - image_size, steps):
            for y in range(ystart, yend - image_size, steps):

                try:  # Fetching diagonals
                    if (ctg1, x) not in diags:
                        diags[(ctg1, x)] = cool.matrix(balance=False).fetch(
                            (ctg1, x * bin_size, (x + image_size) * bin_size),
                            (ctg1, x * bin_size, (x + image_size) * bin_size),
                        )
                    if (ctg2, y) not in diags:
                        diags[(ctg2, y)] = cool.matrix(balance=False).fetch(
                            (ctg2, y * bin_size, (y + image_size) * bin_size),
                            (ctg2, y * bin_size, (y + image_size) * bin_size),
                        )
                except:
                    continue

                ## Fetching matrices using cooler API
                try:
                    arr = whole_mat[x:x + image_size, y:y + image_size]

                    if threshold is None or np.sum(arr) >= threshold:
                        # Reconstructing the full matrix
                        fm = np.zeros((image_size * 2, image_size * 2))
                        ## Adding diagonals
                        fm[0:image_size, 0:image_size] = diags[(ctg1, x)]
                        fm[image_size:, image_size:] = diags[(ctg2, y)]
                        ## Adding adjacency matrix
                        fm[0:image_size, image_size:] = arr
                        fm[image_size:, 0:image_size] = np.rot90( np.flip(arr, axis=1), k=1)

                        # Adding the matrix to the storing list
                        arrays.append(fm)

                        # Adding the opposite corner
                        arrays.append(
                            np.rot90(np.flip(fm, axis=1), -1)
                        )

                        # Adding the name
                        names.append(
                            # Position equals 0 at the start of the contig, 1 at the end.
                            f"{ctg1}:{x * bin_size}-{(x + image_size) * bin_size}|{ctg2}:{y * bin_size}-{(y + image_size) * bin_size}"
                        )
                        names.append(
                            f"{ctg2}:{y * bin_size}-{(y + image_size) * bin_size}|{ctg1}:{x * bin_size}-{(x + image_size) * bin_size}"
                        )

                except Exception as inst:
                    # Skipping if unable to retrieve the matrix (may be due to too short contigs)
                    print(inst)
                    pass

    if len(arrays):  # If even one matrix has been imported
        return arrays, names

    else:  # If no matrix has been imported
        return None, None

def get_whole_matrices(
        cool: cooler.Cooler,
        chrom_names: list
    ):
    """
    Read the whole matrices from the COOL file.

    Inputs:
        - cool: cooler.Cooler. HiC matrix (MCOOL) imported using cooler api.
        - chrom_names: list. List of Contigs present in the cool file.

    Output:
        - dict. Keys are contig names, values are whole HiC matrix.
    """
    whole_matrices = {
        contig: np.array(cool.matrix(balance=False).fetch((contig, None, None), (contig, None, None)))
        for contig in chrom_names
    }

    return whole_matrices

#% Creating the dataset ------------------------------------------------------------------------------------------------
def save_arr(directory, x, names):
    """
    Saving arrays to disk.
    
    """
    test = True

    # Finding unique name
    while test:
        tmp_name = "".join(np.random.choice(list(string.ascii_uppercase), size=16, replace=True))

        test = (f"{tmp_name}.x.npy" in os.listdir(directory))

    # Writing arrays
    np.save(
        os.path.join(directory, f"{tmp_name}.x.npy"),
        x
    )
    np.save(
        os.path.join(directory, f"{tmp_name}.names.npy"),
        names
    )

    # Returning tmp_name
    return tmp_name

def create_inter_contig_dataset(
        bin_size: int,
        image_size: int,
        mat: cooler.Cooler.matrix,
        contig_pairs: list,
        chrom_sizes: dict,
        base_contigs: dict,
        intra_quadrants: dict=None,
        offsets: dict=None,
        threshold: int=None,
        tqdm_params: dict={"desc":"Importing inter-contigs ...", "pos":0},
        hotspot: bool=True,
        tmp_dir: str=".",
        steps: int=1,
        padding: int=-1
    ):
    """
    Create the dataset of inter-contig matrices.

    Inputs:
        - bin_size: int. Genomic range of the matrix bin (or pixel).
        - image_size: int. Size of the HiC matrix.
        - cool: cooler.Cooler. HiC matrix (MCOOL) imported using cooler api.
        - contig_pairs: list. Pairs of contigs to get interaction matrix from.
        - threshold: int. Filter on minimal counts in matrix (default = None).
        - hotspot: bool. Use or not the hotspot search to focus matrices imports (default = True).
        - steps: int. Step between matrices (default = True).
        - tmp_dir: str. Directory where arrays are temporary stored (default = '.').

    Output:
        - dict. Data dict with x, y, names and groups keys.
    """
    data = {}
    all_x, all_names, all_groups, filenames = [], [], [], []
    group_max = 0

    # Write buffers
    BATCH = 12000
    buffer_x = []
    buffer_names = []

    for i in tqdm(range(len(contig_pairs)), desc=tqdm_params["desc"], unit="ctg", position=tqdm_params["pos"], leave=False):
        contig1, contig2 = contig_pairs[i]

        # Fetching matrices
        if hotspot: # Importing hotspot patches
            x, names = import_inter_matrices_hotspot(
                image_size=image_size,
                bin_size=bin_size,
                mat=mat,
                ctg1=contig1,
                ctg2=contig2,
                threshold=threshold,
                steps=steps
            )

        else: # Importing sampled positions
            x, names = import_inter_matrices(
                contig1=contig1,
                contig2=contig2,
                intra_quadrants=intra_quadrants,
                mat=mat,
                offsets=offsets,
                image_size=image_size,
                bin_size=bin_size,
                threshold=threshold,
                padding=padding,
                chrom_sizes=chrom_sizes,
                base_contigs=base_contigs
            )

        # Appending all_x if not empty
        if x is not None:
            assert len(x) == len(names)
            buffer_x.extend(x)
            buffer_names.extend(names)

            if len(buffer_x) >= BATCH or i == len(contig_pairs)-1:
                filename = save_arr(tmp_dir, buffer_x, buffer_names)
                filenames.append(filename)
                buffer_x.clear()
                buffer_names.clear()

            all_groups += [group_max]*len(x)
            group_max += 1

    # Reading temporary arrays
    for filename in filenames:
        all_x += [np.load(os.path.join(tmp_dir, f"{filename}.x.npy"))]
        all_names += [np.load(os.path.join(tmp_dir, f"{filename}.names.npy"))]

        os.remove(os.path.join(tmp_dir, f"{filename}.x.npy"))
        os.remove(os.path.join(tmp_dir, f"{filename}.names.npy"))
    
    # Adding all_x and all_names to data dict
    if len(all_x) >= 1:
        data[(image_size, bin_size)] = {
            "x": np.concatenate(all_x, axis=0),
            "y": np.array([0 for k in all_x for i in range(len(k))]),
            "groups": np.array(all_groups),
            "names": np.concatenate(all_names, axis=0)
        }
    else: data = None

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
        - output: str. Path to the HDF5 output file.
        - whole: dict. Dictionary created using get_whole_matrices().
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

def create_contig_list(
        chrom_sizes: dict,
        bin_size: int,
        image_size: int,
        steps: int=None,
        size_offset: int=6
    ):
    """
    Create contigs based on contigs listed in MCOOL file.

    Inputs:
        - chrom_sizes: dict. Keys are contig name, values are contig length
        - bin_size: int. Genomic range of a bin.
        - image_size: int. Size of the dataset samples.
        - steps: int. Number of steps when importing full dataset.

    Output:
        - list. Sub-contig names such as <contig_name>:<start_bin>-<end_bin>
    """
    sub_contigs = []

    for chrom_name, length in chrom_sizes.items():
        if (length//bin_size) < image_size-size_offset: # Checking if the contig is actually long enough
            print(f"Unable to use {chrom_name}: length {length} < minimum contig size !")
            continue

        # Getting the bin range
        if steps is not None: # If steps is provided
            bin_range = range(0, (length // bin_size), steps)
        else: # If steps is not provided : only the contigs limits
            bin_range = [0, (length // bin_size)]

        sub_contigs += [
            f"{chrom_name}:{bin_range[k]}-{bin_range[k+1]}"
            for k in range(len(bin_range)-1)
        ]

    assert len(sub_contigs) == len(np.unique(sub_contigs))
    return sub_contigs

def decompose_ctg_name(contig):
    chrom, crange = contig.split(":")[:2]
    start, end = crange.split("-")

    return chrom, start, end

def is_contiguous(contigA, contigB):
    chromA, startA, endA = decompose_ctg_name(contigA)
    chromB, startB, endB = decompose_ctg_name(contigB)

    if startB == endA or startA == endB:
        return True
    return False

def main(
        cool_file: str,
        image_size: int,
        bin_size: int,
        output: str,
        threshold: int=None,
        chrom_file: str=None,
        contig_file: str=None,
        threads: int=1,
        steps: int=None,
        hotspot: bool=True,
        tmp_dir: str=None,
        inference: bool=False,
        padding: int=-1
    ):
    """
    Main function to create the dataset, normalize it and output it to HDF5

    Inputs:
        - cool_file: str. Path to the MCOOL.
        - image_size: int. Size of the image to import.
        - bin_size: int. Genomic range of HiC bin.
        - contig_file: str. Path to the corrected contig list.
        - threshold: int. Filter on minimal counts in matrix (default = None).
        - output: str. Path to the output HDF5.
        - threads: int. Number of threads for multiprocessing.
        - steps: int. Number of steps when importing full dataset.
    """
    print("Converting cool file to DLScaff scaffolding dataset")

    # Getting other variables from inputs
    dataset_name = os.path.splitext(os.path.basename(cool_file))[0]

    # Reading COOL file
    try:
        cool = cooler.Cooler(cool_file + f"::/resolutions/{bin_size}")
    except Exception as E1:
        print(f"Unable to open {cool_file}::/resolutions/{bin_size}:\n{E1}\nAttempting to load without specific resolution.")
        try:
            cool = cooler.Cooler(cool_file)
        except Exception as E2:
            print(f"Unable to load :\n{E2}")

    with open(chrom_file, 'r') as handle:
        chrom_sizes = {}
        for line in handle:
            ctg, length = line.rstrip().split("\t")
            chrom_sizes[ctg] = int(length)

    # Importing contigs
    if contig_file is None:
        base_contigs = None
        if hotspot: # Using full Contigs
            contigs = [
                k for k in list(chrom_sizes.keys())
                if int(chrom_sizes[k]) >= image_size*bin_size
            ]

        else: # Creating sub positions
            contigs = create_contig_list(
                chrom_sizes=chrom_sizes,
                image_size=image_size,
                bin_size=bin_size,
                steps = steps
            )
    else:
        contigs, base_contigs = parse_corrected_contigs(
            contig_file=contig_file,
            image_size = image_size,
            bin_size = bin_size,
            chrom_sizes=chrom_sizes
        )

    # Making contig pairs
    if inference:
        contig_pairs = [
            (contigs[idA], contigs[idB])
            for idA in range(len(contigs) - 1)
            for idB in range(idA, len(contigs))
            if contigs[idA] != contigs[idB]
        ]
    else:
        contig_pairs = [
            (contigs[idA], contigs[idB])
            for idA in range(len(contigs)-1)
            for idB in range(idA, len(contigs))
            if contigs[idA] != contigs[idB]
            and (hotspot or steps is None or not is_contiguous(contigs[idA], contigs[idB]))
        ]

    # Debug print
    #print(contig_pairs)

    # Debug check
    assert len(np.unique(np.unique([f"{k1}|{k2}" for k1, k2 in contig_pairs]))) == len(contig_pairs)

    # Assigning tmp_dir
    if tmp_dir is None:
        tmp_dir=output

    mat = cool.matrix(balance=False, sparse=True)[:].tocsr()

    offsets = {ctg.split(":")[0]: cool.offset(ctg.split(":")[0]) for ctg in contigs}
    intra_quadrants = parse_intra_quadrants(contigs=contigs, image_size=image_size, mat=mat, offsets=offsets) if not hotspot else None
    #inter_quadrants = parse_inter_quadrants(contig_pairs=contig_pairs, image_size=image_size, mat=mat, offsets=offsets) if not hotspot else None

    # Creating the dataset
    if threads == 1:
        data = create_inter_contig_dataset(
            bin_size=bin_size,
            image_size=image_size,
            mat=mat,
            threshold=threshold,
            intra_quadrants=intra_quadrants,
            offsets=offsets,
            contig_pairs=contig_pairs,
            hotspot=hotspot,
            tmp_dir=tmp_dir,
            steps=steps,
            padding=padding,
            chrom_sizes=chrom_sizes,
            base_contigs=base_contigs
        )

    else:
        # Starting multiprocessing
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            splits = np.array_split(contig_pairs, threads)
            tqdm_base_pos = tqdm._get_free_pos()
            for sub_list_id in range(len(splits)):
                os.makedirs(os.path.join(tmp_dir, f"split_{sub_list_id}"), exist_ok=True)
                results.append(
                    executor.submit(
                        create_inter_contig_dataset,
                        bin_size,
                        image_size,
                        mat,
                        splits[sub_list_id].tolist(),
                        chrom_sizes,
                        base_contigs,
                        intra_quadrants,
                        offsets,
                        threshold,
                        {"desc":f"Batch {sub_list_id}/{threads}", "pos":tqdm_base_pos+sub_list_id},
                        hotspot,
                        os.path.join(tmp_dir, f"split_{sub_list_id}"),
                        steps,
                        padding
                    )
                )

        # Aggregating results
        data = {(image_size, bin_size): {"x":[], "y":[], "groups":[], "names":[]}}
        for raw_res in results:
            for key, value in raw_res.result()[(image_size, bin_size)].items():
                data[(image_size, bin_size)][key].append(value)

        # Stacking results
        for key, value in data[(image_size, bin_size)].items():
            data[(image_size, bin_size)][key] = np.concatenate(value, axis=0)

    # Fetching whole matrices
    whole_matrices = get_whole_matrices(
        cool=cool,
        chrom_names=list(cool.chromsizes.to_dict().keys())
    )

    # Sanity checks
    for key in data.keys():
        assert data[key]["x"].shape[0] == data[key]["y"].shape[0]
        assert data[key]["x"].shape[0] == data[key]["groups"].shape[0]
        assert data[key]["x"].shape[0] == data[key]["names"].shape[0]

    # Exporting to HDF5
    metadata = {
        "Cool2InterM_FullMat.version": version,
        "data.type": "Inter.FullMat",
        "dataset": dataset_name,
        "Cool2InterM_FullMat.filter": threshold if threshold is not None else "None",
        "Cool2InterM_FullMat.creation_date": datetime.datetime.now().strftime("%m/%d/%Y-%H:%M:%S")
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
        default = 50,
        help = "Sizes of images (Default: 50)"
        )
    argParser.add_argument(
        "--bin-size",
        "-b",
        type = int,
        dest = "bin_size",
        default = 10000,
        help = "Genomic size of a bin (Default: 10000)"
        )
    argParser.add_argument(
        "--chrom-sizes",
        dest="chrom_file",
        required=True,
        help="Chrom sizes file",
        type=str
    )
    argParser.add_argument(
        "--contigs",
        dest = "contig_file",
        required = False,
        help = "Contig file produced by contig_corrector.py",
        type = str,
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
        "--threshold",
        dest="threshold",
        required=False,
        help="Minimum counts in matrix",
        type=int,
        default=None
    )
    argParser.add_argument(
        "--threads",
        dest="threads",
        required=False,
        help="Number of parallel process",
        type=int,
        default=1
        )
    argParser.add_argument(
        "--steps",
        dest="steps",
        required=False,
        help="Number of steps when importing full dataset (Default is None)",
        type=int,
        default=None
        )
    argParser.add_argument(
        "--no-hotspot",
        dest="hotspot",
        action='store_false',
        help="Disable import matrices in hotspots only."
    )
    argParser.add_argument(
        "--tmp-dir",
        dest="tmp_dir",
        required=False,
        help="Tmp directory (Default is None)",
        type=str,
        default=None
        )
    argParser.add_argument(
        "--inference",
        dest="inference",
        action='store_true',
        help="Generate inference mode dataset."
    )
    argParser.add_argument(
        "--padding",
        dest="padding",
        required=False,
        help="Padding value",
        type=int,
        default=-1
    )
    argParser.add_argument(
        "--profile",
        dest="profile",
        action='store_true',
        help="Generate profile."
    )
    args = argParser.parse_args()

    if args.inference: # Safeguard
        args.hotspot = False
        args.steps = None

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        main(
            cool_file=args.cool_file,
            image_size=args.image_size,
            bin_size=args.bin_size,
            chrom_file=args.chrom_file,
            contig_file=args.contig_file,
            threshold=args.threshold,
            output=args.output,
            threads=args.threads,
            steps=args.steps,
            hotspot=args.hotspot,
            tmp_dir=args.tmp_dir,
            inference=args.inference,
            padding=args.padding
        )
        profiler.disable()
        profiler.dump_stats(os.path.join(os.path.dirname(args.output),"Cool2Inter.prof"))

    else:
        main(
            cool_file = args.cool_file,
            image_size = args.image_size,
            bin_size = args.bin_size,
            chrom_file = args.chrom_file,
            contig_file = args.contig_file,
            threshold = args.threshold,
            output = args.output,
            threads = args.threads,
            steps = args.steps,
            hotspot = args.hotspot,
            tmp_dir = args.tmp_dir,
            inference = args.inference,
            padding = args.padding
        )