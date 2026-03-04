#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DLScaff dataset main method for working with datasets

@author: Alexis Mergez
@version: 5.7
@Last modified: 2026/01/12
"""

import os
import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as plc

from torch.utils.data import Dataset
from tqdm import tqdm
import torch

class dataset():
    
    def __init__(self, path, image_size = 10, bin_size = 5000, skip_dataset=False, skip_whole=True, skip_check=False):
        """
        Dataset class for handling dlscaff datasets with ease.

        Parameters
        ----------
        path : str
            Fullpath to the dataset.
        image_size : int, optional
            Number of bins in image. The default is 10.
        bin_size : int, optional
            Number of bases in a bin. The default is 5000.

        Raises
        ------
        FileNotFoundError
            Path does not exists.

        Returns
        -------
        dataset object.

        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Unable to find {path}")
        
        self.path = path
        self.dir = os.path.dirname(self.path)
        self.image_size = image_size
        self.bin_size = bin_size
        self.isSplitted = False
        
        self.version = "5.7"
        print(f"DLScaff dataset v{self.version}\n")

        self.load(skip_dataset=skip_dataset, skip_whole=skip_whole, skip_check=skip_check)
        
    def load(self, skip_dataset=False, skip_whole=True, skip_check=False):
        """
        Loader for DLScaff dataset.
        Reads a HDF5 file and import it as a dictionnary called self.dataset. 

        Raises
        ------
        ValueError
            Unable to find either bin size or image size.

        Returns
        -------
        None.

        """
        startTime = time.time()

        self.dataset = {}
        
        with h5py.File(self.path, 'r') as inData:
            print(f"Loading {self.path} ...")
            
            # Getting image sizes and checking if selected one (self.image_size) is available
            availimage_sizes = list(inData.keys())
            if not str(self.image_size) in availimage_sizes:
                raise ValueError(f"Unable to find data with image size of {self.image_size}")
            
            # Getting bin size and checking if selected one (self.bin_size) is available
            availbin_sizes = list(inData[str(self.image_size)].keys())
            if not str(self.bin_size) in availbin_sizes:
                raise ValueError(f"Unable to find data with bin size of {self.bin_size}")

            # Getting data and storing it in self.dataset
            if not skip_dataset:        
                self.dataset = {
                    "x" : np.array(inData[str(self.image_size)][str(self.bin_size)]["x"], dtype=np.int32),
                    "y" : np.array(inData[str(self.image_size)][str(self.bin_size)]["y"], dtype = "int"),
                    "names" : np.array(inData[str(self.image_size)][str(self.bin_size)]["names"], dtype = "str"),
                    "groups" : np.array(inData[str(self.image_size)][str(self.bin_size)]["groups"], dtype=np.uint16)
                    }

                # Creating shortcuts
                self.X = self.dataset["x"]
                self.Y = self.dataset["y"]
                self.N = self.dataset["names"]
                self.G = self.dataset["groups"]
            
            # Trying to load full arrays
            if not skip_whole:
                try :
                    self.whole_matrices = {
                        ctg: np.array(matrix) for ctg, matrix in inData[str(self.image_size)][str(self.bin_size)]["whole"].items()
                    }
                except Exception as e:
                    print(e)
                    pass

            self.datasetName = inData.attrs["dataset"]
            self.metadata = {attributes: inData.attrs[attributes] for attributes in inData.attrs.keys()}

        print(f"Done ! ({round(time.time()-startTime,3)}s)")
        
        if not skip_dataset and not skip_check :
            print("Checking if labels are identical within each groups ...")
            for groupid in np.unique(self.dataset["groups"]):
                indices = self.dataset["y"][self.dataset["groups"] == groupid]
                assert len(np.unique(indices)) == 1
            print("Pass !")

        print("\nDataset metadata :")
        for attributes in self.metadata.keys():
            print(f"\t{attributes}: {self.metadata[attributes]}")
        print("")
        
    def filter(self, threshold = .1, absThres = None, dump = True):
        """
        Filter the dataset.

        Parameters
        ----------
        threshold : float, optional
            Percentage of the median to set as threshold. The default is .1.
        absThres : int, optional
            Overrides threshold. Minimal number of HiC counts in a matrix. The default is None.
        dump : bool, optional
            Removes filtered matrices if True. Default is True.

        Returns
        -------
        None.

        """
        
        print("Filtering the dataset ...")
        sumsPerMatrices = np.sum(np.sum(self.dataset["x"], axis = 2), axis = 1)
        
        median = np.quantile(sumsPerMatrices, 0.5)
        
        ## Building a list of bool to select matrices within the array
        selector = []
        for i in range(len(sumsPerMatrices)):
            if absThres is None and sumsPerMatrices[i] <= threshold*median:
                selector.append(False)
                    
            elif absThres is not None and sumsPerMatrices[i] <= absThres:
                selector.append(False)
                
            else : selector.append(True)
            
        selector = np.array(selector, dtype = bool)
        
        print(f"\t{np.sum(selector.astype(int) == 0)} matrices will be removed")
        
        ## Storing away the filtered matrices
        if not dump :
            self.filtered = {
                "x": self.dataset["x"][(selector == 0),],
                "y": self.dataset["y"][(selector == 0),],
                "names": self.dataset["names"][(selector == 0),],
                "groups": self.dataset["groups"][(selector == 0),],
                }
        
        self.dataset["x"] = self.dataset["x"][selector,]
        self.dataset["y"] = self.dataset["y"][selector,]
        self.dataset["names"] = self.dataset["names"][selector,]
        self.dataset["groups"] = self.dataset["groups"][selector,]  
        
    def summary(self):
        """
        Print a summary of the dataset and sub-datasets if available.

        Returns
        -------
        None.

        """
        print(f"'{self.datasetName}' dataset summary - BS:{self.bin_size} IS:{self.image_size}")
        if hasattr(self, "is_normed") and self.is_normed:
            print("\n\tThe dataset has been normed.")

        print(f"\tWhole dataset :")
        for elem in self.dataset.keys():
            print(f"\t\t{elem} - {self.dataset[elem].shape}")
        print(f"\t\t{round((np.sum(self.dataset['y'])/self.dataset['y'].shape[0])*100,2)}% of the samples are True")

        if hasattr(self, "train"):
            print(f"\n\tTrain sub-dataset summary :")
            for elem in self.train.keys():
                print(f"\t\t{elem} - {self.train[elem].shape}")
            print(f"\t\t{round((np.sum(self.train['y'])/self.train['y'].shape[0])*100,2)}% of the samples are True")
            
        if hasattr(self, "test"):
            print(f"\n\tTest sub-dataset summary :")
            for elem in self.test.keys():
                print(f"\t\t{elem} - {self.test[elem].shape}")
            print(f"\t\t{round((np.sum(self.test['y'])/self.test['y'].shape[0])*100,2)}% of the samples are True")
            
        if hasattr(self, "val"):
            print(f"\n\tVal sub-dataset summary :")
            for elem in self.val.keys():
                print(f"\t\t{elem} - {self.val[elem].shape}")
            print(f"\t\t{round((np.sum(self.val['y'])/self.val['y'].shape[0])*100,2)}% of the samples are True")

        if hasattr(self, "c2"):
            print(f"\n\tC2 sub-dataset summary :")
            for elem in self.c2.keys():
                print(f"\t\t{elem} - {self.c2[elem].shape}")
            print(f"\t\t{round((np.sum(self.c2['y'])/self.c2['y'].shape[0])*100,2)}% of the samples are True")

        if hasattr(self, "c3"):
            print(f"\n\tC3 sub-dataset summary :")
            for elem in self.c3.keys():
                print(f"\t\t{elem} - {self.c3[elem].shape}")
            print(f"\t\t{round((np.sum(self.c3['y'])/self.c3['y'].shape[0])*100,2)}% of the samples are True")
            
    def fullCheck(self):
        """
        Search for similar matrices within the whole dataset. 
        Very computational heavy task and may take days.
        Results are saved within the dataset folder in fullCheck.logs.

        Returns
        -------
        None.

        """
        print("Checking if a matrice is duplicated ...")
        nchecksToDo = int(len(self.dataset["x"])*(len(self.dataset["x"])-1)*.5)
        iteration = 0
        
        with open(os.path.join(self.dir, "fullCheck.logs"), "a") as file :
            file.write(f"[DLScaffDataset]\tfullCheck logs\t{time.asctime()}\n")
            
        for A in range(len(self.dataset["x"])-1):
            for B in range(A+1, len(self.dataset["x"])):
                if np.all(self.dataset["x"][A] == self.dataset["x"][B]):
                    with open(os.path.join(self.dir, "fullCheck.logs"), "a") as file :
                        file.write(f"{A}:{self.dataset['names'][A]} is the same as {B}:{self.dataset['names'][B]}\n")
                        
                iteration += 1
                
                if iteration%100000 == 0:
                    print(f"{iteration} out of {nchecksToDo} ({round((iteration/nchecksToDo)*100, 2)}%)")

    def get_vmax(self):
        if not hasattr(self, "vmax"):
            self.vmax = max(np.quantile(self.dataset["x"], 0.995), 1)
        return self.vmax

    def show(self, save = False, fontsize = 7, alpha=1, abs_vmax=True, log1p=True):
        """
        Take 9 random samples in the dataset and print it as a figure.

        Parameters
        ----------
        save : bool, optional
            Toggle figure saving as RandomSample.png in the dataset directory. The default is False.

        Returns
        -------
        None.

        """
        selectedIndices = np.random.choice(
            self.dataset["x"].shape[0],
            size = 9,
            replace = False
            )
        
        fig = plt.figure(figsize=(10, 10), dpi = 300)
        fig.subplots_adjust(wspace=0.1, hspace=0.12, top=0.97, left=0, right=1, bottom=0)
        fig.patch.set_facecolor((1,1,1,alpha))
        plt.rcParams['axes.titleweight'] = "bold"

        for k in range(9): 
            if abs_vmax: vmax = self.get_vmax()
            else :
                try :
                    ctg_name = "_".join( self.dataset['names'][selectedIndices[k]].split("_")[:-2] )
                    vmax = max(np.quantile(self.whole_matrices[ctg_name], 0.999), 1)
                except :
                    vmax = self.get_vmax()

            ax = plt.subplot(3, 3, k+1)
            ax.imshow(
                self.dataset["x"][selectedIndices[k]],
                cmap = "hot_r",
                norm = plc.AsinhNorm(vmin = 0, vmax = vmax),
                aspect = 'equal')
            ax.set_title(
                f"{self.datasetName} - {self.dataset['names'][selectedIndices[k]]} - {self.dataset['y'][selectedIndices[k]]}",
                fontsize = fontsize
                )
            plt.axis('off')
            
        if save : plt.savefig(
                os.path.join(self.dir, "RandomSample.png")
                )
    
    def whole_names(self):
        if not hasattr(self, "whole_matrices"):
            raise ValueError("No whole matrix is available in this dataset !")
        return list(self.whole_matrices.keys())

    def list_whole(self):
        if not hasattr(self, "whole_matrices"):
            raise ValueError("No whole matrix is available in this dataset !")

        print("List of whole matrix names :")
        for ctg, matrix in self.whole_matrices.items():
            print(f"\t{ctg}: matrix of size {matrix.shape}")

    def show_whole(self, name, save = None, fontsize = 7, alpha=1):
        if not hasattr(self, "whole_matrices"):
            raise ValueError("No whole matrix is available in this dataset !")

        fig = plt.figure(figsize=(5, 5), dpi = 300)
        fig.subplots_adjust(wspace=0.1, hspace=0.12, top=0.97, left=0, right=1, bottom=0)
        fig.patch.set_facecolor((1,1,1,alpha))
        plt.rcParams['axes.titleweight'] = "bold"

        ax = plt.subplot(1, 1, 1)
        ax.imshow(
            self.whole_matrices[name],
            cmap = "hot_r",
            norm = plc.AsinhNorm(vmin = 0, vmax = max(self.get_vmax(), 1)),
            aspect = 'equal')
        ax.set_title(
            f"{self.datasetName} - {name}",
            fontsize = fontsize
            )
        plt.axis('off')

        if save is not None:
            plt.savefig(save)

    def sample(self, name, alpha=1):
        """
        Getting the array with its name.

        Parameters
        ----------
        name : str
            Name of the array.

        Returns
        -------
        None.

        """
        fig = plt.figure(figsize=(5, 5), dpi = 300)
        fig.subplots_adjust(wspace=0.1, hspace=0.12, top=0.97, left=0, right=1, bottom=0)
        fig.patch.set_facecolor((1,1,1,alpha))
        plt.rcParams['axes.titleweight'] = "bold"
        
        selector = np.isin(self.dataset["names"], name)
        
        ax = plt.subplot(1, 1, 1)
        ax.imshow(
            self.dataset["x"][selector][0],
            cmap = "hot_r",
            norm = plc.AsinhNorm(vmin = 0, vmax = np.max(self.dataset["x"][selector][0])),
            aspect = 'equal')
        ax.set_title(
            f"{self.datasetName} - {name} - {self.dataset['y'][selector][0]}",
            fontsize = 7
            )
        plt.axis('off')
    
    def samples(self, names, fontsize = 7, save = False, alpha=1):
        """
        Take 9 random samples in a given list of names and print it as a figure.

        Parameters
        ----------
        names : np.array
            Names of arrays
        save : bool, optional
            Toggle figure saving as Samples.png in the dataset directory. The default is False.

        Returns
        -------
        None.

        """
        selectedIndices = np.random.choice(
            np.where(np.isin(self.dataset["names"], names))[0],
            size = min(len(names), 9),
            replace = False
            )
        
        fig = plt.figure(figsize=(10, 10), dpi = 300)
        fig.subplots_adjust(wspace=0.1, hspace=0.12, top=0.97, left=0, right=1, bottom=0)
        fig.patch.set_facecolor((1,1,1,alpha))
        plt.rcParams['axes.titleweight'] = "bold"

        for k in range(min(len(names), 9)): 
            ax = plt.subplot(3, 3, k+1)
            ax.imshow(
                self.dataset["x"][selectedIndices[k]],
                cmap = "hot_r",
                norm = plc.AsinhNorm(vmin = 0, vmax = self.get_vmax()),
                aspect = 'equal')
            ax.set_title(
                f"{self.datasetName} - {self.dataset['names'][selectedIndices[k]]} - {self.dataset['y'][selectedIndices[k]]}",
                fontsize = fontsize
                )
            plt.axis('off')
            
        if save : plt.savefig(
                os.path.join(self.dir, "Samples.png")
                )

    def search_sample(self, keyA, keyB=None, type="and"):
        lower_names = np.char.lower(self.dataset["names"])
        maskA = np.char.find(lower_names, np.char.lower(keyA)) != -1
        if keyB is not None:
            maskB = np.char.find(lower_names, np.char.lower(keyB)) != -1
            mask = maskA & maskB
        else:
            mask = maskA

        self.samples(self.dataset["names"][mask])

    def dist(self, nbins = 100, limits = [.2, .4, .5, .6, .8], to_log=False, savedir=None, max=None, title=None):
        """
        Plot an histogram showing the distribution of sums of counts in matrices.

        Parameters
        ----------
        nbins : TYPE, optional
            DESCRIPTION. The default is 100.
        limits : TYPE, optional
            DESCRIPTION. The default is [.2, .4, .5, .6, .8].

        Returns
        -------
        None.

        """
        
        sumsPerMatrices = np.sum(np.sum(self.dataset["x"], axis = 2), axis = 1)

        fig, ax = plt.subplots(figsize = (10, 3), dpi = 300)

        if max is None:
            ax.hist(sumsPerMatrices, bins = 100, log=to_log)
        else:
            ax.hist(sumsPerMatrices, bins=100, log=to_log, range=[0, max])

        quantiles = [np.quantile(sumsPerMatrices, q) for q in limits]
        
        for q in quantiles:
            ax.axvline(q, color = "black", ls="--")
            
        ax.axvline(np.quantile(sumsPerMatrices, 0.5)*0.1, color = "red", ls="--")
        ax.set_xlabel("Total number of contacts in matrix")
        ax.set_ylabel("Count")

        if title is not None:
            ax.set_title(title, fontweight='bold')

        plt.tight_layout()
        if savedir is not None:
            plt.savefig(savedir)
        plt.show()

    def _split_engine(self, class_indices, val_size, test_size, seed):
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size+test_size, random_state=seed)
        train_idx, test_val_idx = next(splitter.split(
            self.dataset["x"][class_indices],
            self.dataset["y"][class_indices],
            groups=self.dataset["groups"][class_indices])
        )
        adjusted_val_size = val_size / (val_size+test_size)
        splitter_val = GroupShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=seed)
        test_idx, val_idx = next(splitter_val.split(
            self.dataset["x"][class_indices[test_val_idx]],
            self.dataset["y"][class_indices[test_val_idx]],
            groups=self.dataset["groups"][class_indices[test_val_idx]])
        )

        return class_indices[train_idx], class_indices[test_val_idx[val_idx]], class_indices[test_val_idx[test_idx]]

    def split(self, test_size=0.05, val_size=0.05, seed=42):

        from sklearn.model_selection import GroupShuffleSplit

        np.random.seed(seed)

        # Getting classes indices
        class_0_indices = np.where((self.dataset["y"] == 0))[0]
        class_1_indices = np.where((self.dataset["y"] == 1))[0]

        # Splitting on class_1
        train_c1_idx, val_c1_idx, test_c1_idx = self._split_engine(
            class_indices=class_1_indices,
            val_size=val_size,
            test_size=test_size,
            seed=seed
        )

        # Splitting on class_0
        train_c0_idx, val_c0_idx, test_c0_idx = self._split_engine(
            class_indices=class_0_indices,
            val_size=val_size,
            test_size=test_size,
            seed=seed
        )

        self.train = {
            elem: self.dataset[elem][np.concatenate([
                train_c1_idx,
                train_c0_idx
            ])] for elem in self.dataset.keys()
        }
        self.val = {
            elem: self.dataset[elem][np.concatenate([
                val_c1_idx,
                val_c0_idx
            ])] for elem in self.dataset.keys()
        }
        self.test = {
            elem: self.dataset[elem][np.concatenate([
                test_c1_idx,
                test_c0_idx
            ])] for elem in self.dataset.keys()
        }

        self.isSplitted = True

    def split_legacy(self, splitTest=None, splitVal=None, seed=None):
        """
        **DEPRECATED**
        Split the dataset in wanted subsets.

        Parameters
        ----------
        splitTest : float, optional
            Test subset proportion of the dataset. None means no 'test' subset will be created.
            The default is None.
        splitVal : float, optional
            Val subset proportion of the dataset. None means no 'val' subset will be created.
            The default is None.
        seed : int, optional
            Set the seed for reproducible split

        Returns
        -------
        None.

        """
        self.isSplitted = True

        if seed is not None: rng = np.random.default_rng(seed=seed)
        else : rng = np.random.default_rng()

        if splitTest is not None or splitVal is not None:
            groups = np.unique(self.dataset["groups"])
            nGroups = len(groups)
            
            ## Splitting groups in each sets
            if splitTest is not None : 
                elementsInTest = int(splitTest*nGroups)+1
                selectedTest = rng.choice(
                    groups,
                    size = elementsInTest,
                    replace = False
                )

                testIndices = np.isin(self.dataset["groups"], selectedTest)
                groups = np.setdiff1d(groups, selectedTest)
            
            if splitVal is not None :
                elementsInVal = int(splitVal*nGroups)+1
                selectedVal = rng.choice(
                    groups, 
                    size = elementsInVal, 
                    replace = False
                )
                
                valIndices = np.isin(self.dataset["groups"], selectedVal)
                groups = np.setdiff1d(groups, selectedVal)
            
            if splitVal is not None and splitTest is not None:
                assert np.intersect1d(selectedTest, selectedVal).size == 0
                assert not np.array_equal(testIndices, valIndices)
            
            trainIndices = np.isin(self.dataset["groups"], groups)
        
        
            self.train = {
                elem: self.dataset[elem][trainIndices,] for elem in self.dataset.keys()
                }
            
            if splitTest is not None :
                self.test = {
                    elem: self.dataset[elem][testIndices,] for elem in self.dataset.keys()
                    }
            if splitVal is not None :
                self.val = {
                    elem: self.dataset[elem][valIndices,] for elem in self.dataset.keys()
                    }
                
        else : self.train = self.dataset
        
    def get(self, subset = "all"):
        """
        Return the desired subset.

        Parameters
        ----------
        subset : str, optional
            -"all" for the whole dataset. 
            -"train" for train subset.
            -"test" for test subset.
            -"val" for validation subset.
            -"c2" or "c3" for corner 2 or corner 3 full dataset
            The default is "all".

        Raises
        ------
        AttributeError
            The desired subset is not available.

        Returns
        -------
        dict
            Dataset as a dict with following structure:
                -"x"
                -"y"
                -"groups"
                -"names"
                
        """
        if subset == "all": subset = "dataset"
        if not hasattr(self, subset):
            raise AttributeError(f"'{subset}' subset does not exist")
            
        if subset == "dataset":
            return self.dataset
        if subset == "train":
            return self.train
        if subset == "test":
            return self.test
        if subset == "val":
            return self.val
        if subset == "c2":
            return self.c2
        if subset == "c3":
            return self.c3

    def get_as_torch(self, subset = "all", transform = None, target_transform=None):
        return torch_dataset(self.get(subset), transform, target_transform)
        
    def checkSubset(self):
        """
        Checks if groups or names are shared between subsets.

        Returns
        -------
        None.

        """
        
        print("Checking if a group is in multiple subset ...")
        chkGroups = sum([
            len(set(self.test["groups"]) & set(self.train["groups"])),
            len(set(self.test["groups"]) & set(self.val["groups"])),
            len(set(self.train["groups"]) & set(self.val["groups"]))
            ])
        assert chkGroups == 0
        
        print("Checking if a name is in multiple subset ...")
        chkNames = sum([
            len(set(self.test["names"]) & set(self.train["names"])),
            len(set(self.test["names"]) & set(self.val["names"])),
            len(set(self.train["names"]) & set(self.val["names"]))
            ])
        assert chkNames == 0
            
    def shape(self):
        """
        Give the shape of an image. Used when creating the model.

        Returns
        -------
        tuple
            Matrix shape.

        """
        return (self.dataset["x"].shape[1], self.dataset["x"].shape[2], 1)

    def save(self, savedir):
        with h5py.File(savedir, "w") as out:
        
            print(f"\tWriting dataset to {savedir}")
            imgSizeGroup = out.create_group(str(self.image_size))
                
            print(f"\tAdding BS={self.bin_size} to IS={self.image_size} group")
            bin_sizeGroup = imgSizeGroup.create_group(str(self.bin_size))
                    
            for key in self.dataset.keys():
                
                print(f"\tAdding {key} to {self.image_size}:{self.bin_size} group")
                
                if key == "names" :
                    bin_sizeGroup.create_dataset(
                        key, 
                        data=self.dataset[key].astype('S'),
                        compression = "gzip"
                    )

                else :
                    bin_sizeGroup.create_dataset(
                        key, 
                        data=self.dataset[key],
                        compression = "gzip"
                        )

            if hasattr(self, "whole_matrices"):
                wholeGroup = bin_sizeGroup.create_group("whole")

                for ctg, whole_matrix in self.whole_matrices.items():
                    wholeGroup.create_dataset(
                        ctg,
                        data = whole_matrix,
                        compression="gzip"
                    )

            for key, value in self.metadata.items():
                out.attrs[key] = value
            
            out.attrs["Edit"] = f"Modified with DLScaffDataset v{self.version}"

        print(f"Done !")

    def norm(self, mean, std, min_, max_):
        """
        Log1P, then mean-centered and standardized, then scaled in [0, 1].
        """
        # Log1P
        self.dataset["x"] = np.log1p(self.dataset["x"])

        # Mean-centered and standardized
        self.dataset["x"] -= mean
        self.dataset["x"] /= std

        # Scaled between 0 and 1
        self.dataset["x"] -= min_
        self.dataset["x"] /= (max_ - min_)

        self.is_normed = True

    def get_quantile_full(self, quantile=.999, low_ram=True, return_value=False):
        values = []
        if not low_ram:
            try:
                for arr in self.whole_matrices.values():
                    values.extend(arr.flatten())
            except:
                print("Unable to load whole matrices !")

        if low_ram: # Directly parsing the file
            with h5py.File(self.path, 'r') as inData:
                # First parse to get the shapes and compute the number of pixels
                n_values = 0
                for matrix in tqdm(inData[str(self.image_size)][str(self.bin_size)]["whole"].values()):
                    n_values += np.array(matrix).size

                # Second parse to fill the array
                values = np.zeros(n_values, dtype=np.int16)
                last_pos=0
                for matrix in tqdm(inData[str(self.image_size)][str(self.bin_size)]["whole"].values()):
                    _ = np.array(matrix).flatten().astype(np.int16)
                    values[last_pos:last_pos+_.size] = _
                    last_pos += _.size
                    del _

        res = np.quantile(values[values != 0], q=quantile)

        if not return_value:
            del values
            return res
        else:
            return res, values

    def get_quantile_diag(self, quantile=.95, return_value=False):
        values = []

        with h5py.File(self.path, 'r') as inData:
            for matrix in tqdm(inData[str(self.image_size)][str(self.bin_size)]["whole"].values(), unit="contig", desc="Fetching diagonals values"):
                _ = np.array(matrix)

                for k in range(-self.image_size, self.image_size+1):
                    values.extend(_.diagonal(k))

        res = np.quantile(values, q=quantile)
        self.q95 = res

        if not return_value:
            return res
        else:
            return res, values

    def minmax(self, max_val=None, quantile=.95, padding=-1):
        """
        MinMax using the given quantile as the upper limit (1) an clipping anything higher
        than this quantile to its value.

        """
        if max_val is None:
            max_val = self.get_quantile_diag(quantile)

        mask = (self.dataset["x"] == padding)

        self.dataset["x"] = self.dataset["x"].astype(np.float32)
        np.divide(self.dataset["x"], max_val, out=self.dataset["x"])
        #self.dataset["x"] = self.dataset["x"].astype(np.float32) / max_val
        np.clip(self.dataset["x"], 0, 1, out=self.dataset["x"])

        self.dataset["x"][mask] = padding

        del mask

        self.is_normed = True

    def get_whole_stats(self, log=True):
        """
        Returns mean, std, min, max for whole matrices.
        """

        means, stds, mins, maxs, sizes = [], [], [], [], []

        # Computing Mean, and STD for whole dataset
        for oc_matrix in self.whole_matrices.values():
            if log:
                matrix = np.log1p(oc_matrix)
            else:
                matrix = oc_matrix
            
            means.append(np.mean(matrix))
            stds.append(np.std(matrix))
            sizes.append(matrix.size)

        mean = np.sum(np.array(means) * np.array(sizes)) / np.sum(sizes)
        var = np.sum(sizes * (np.array(stds)**2 + (np.array(means)- mean)**2)) / np.sum(sizes)
        std = np.sqrt(var)

        # Applying normalization and computing min and max
        for oc_matrix in self.whole_matrices.values():
            if log:
                matrix = np.log1p(oc_matrix)
            else:
                matrix = oc_matrix

            matrix -= mean
            matrix /= std

            mins.append(np.min(matrix))
            maxs.append(np.max(matrix))

        min_ = np.min(mins)
        max_ = np.max(maxs)
        
        return mean, std, min_, max_

    def fix_groups(self):
        """
        Method to fix the groups in case we have merged datasets from the same contigs but with different Hi-C reads.
        In that case, data is equivalent an can be considered as an augmentation instead of a true different sample.
        """
        for name in np.unique(self.dataset["names"]):
            selector = (self.dataset["names"] == name)

            group_ids = np.unique(self.dataset["groups"][selector])
            
            self.dataset["groups"][selector] = min(group_ids)

    def get_groups_type(self):
        """
        Creates 2 selectors for intra and inter contigs.
        """
        # Selecting based on the name
        self.inter_contigs = np.array([
            "|" in name
            for name in self.dataset["names"]
        ])
        self.intra_contigs = self.inter_contigs == False

        # Getting groups id
        self.inter_groups = np.unique(self.dataset["groups"][self.inter_contigs,])
        self.intra_groups = np.unique(self.dataset["groups"][self.intra_contigs,])

    def get_misclassified(self, selector, transform = None, target_transform=None):
        """
        Method to retrieve wrongly predicted samples to resume training of models
        """
        self.misclassified = {
            elem: self.dataset[elem][selector,] for elem in self.dataset.keys()
        }

        return torch_dataset(self.misclassified, transform, target_transform)

    def split_by_corner(self):
        """
        Method to get one dataset per corner to prevent having the 2 opposite corners in the same batch.

        """
        # Get a bool array of whether the matrix is corner 2 or not
        selector_c2 = np.array(
            [k.split("_")[-1] == "C2" for k in self.dataset["names"]]
        )

        self.c2 = {
            elem: self.dataset[elem][selector_c2,] for elem in self.dataset.keys()
        }

        selector_c3 = selector_c2 == False

        self.c3 = {
            elem: self.dataset[elem][selector_c3,] for elem in self.dataset.keys()
        }

    def equalize(self, clip_value=1e3, n_bins=100, seed=None):
        """
        Equalize the number of elements by bins by clipping bins with too many representants.

        """

        if seed is not None: rng = np.random.default_rng(seed=seed)
        else : rng = np.random.default_rng()

        # Computing the sum of pixels values in each matrix
        sums_per_mat = np.sum(np.sum(self.dataset["x"], axis=2), axis=1)

        # Binning in 100 bins and getting an array giving the bin number of each matrix
        _, bin_edges = np.histogram(sums_per_mat, bins=n_bins)
        bin_indices = np.digitize(sums_per_mat, bin_edges) - 1

        # Clipping bin
        selected_indices = []
        for bin_idx in range(n_bins):
            # Getting the indices of matrix belonging to the bin group
            indices_in_bin = np.where(bin_indices == bin_idx)[0]
            count = len(indices_in_bin)

            if count > int(clip_value):
                selected_indices.extend(
                    rng.choice(indices_in_bin, int(clip_value), replace=False)
                )

            else:
                selected_indices.extend(indices_in_bin)

        # Filtering the dataset
        self.dataset["x"] = self.dataset["x"][selected_indices,]
        self.dataset["y"] = self.dataset["y"][selected_indices,]
        self.dataset["names"] = self.dataset["names"][selected_indices,]
        self.dataset["groups"] = self.dataset["groups"][selected_indices,]

    def resize(self, new_image_size):
        """
        Resize the image to the new image size.

        """
        self.dataset["x"] = self.dataset["x"][:, :new_image_size, :new_image_size]

class torch_dataset(Dataset):

    def __init__(self, dls_dataset, transform = None, target_transform=None):

        self.data = dls_dataset

        self.transform = transform
        self.target_transform = target_transform   

    def __len__(self):
        return len(self.data["y"])

    def __getitem__(self, idx):

        image = torch.tensor(self.data["x"][idx][np.newaxis,:,:], dtype=torch.float32) # Adding the channel axis for compatibility reasons
        label = torch.tensor(self.data["y"][idx], dtype=torch.int8)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)       
        
        return image, label

