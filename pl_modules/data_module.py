import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import os
from typing import Union, Optional, Callable
import h5py
from fastmri.data.mri_data import FastMRIRawDataSample
from modules.transforms import LDMSample
import pickle
import torch
from torch.utils.data import DistributedSampler

class LatentDataset(Dataset):
    def __init__(
            self,
            root: Union[str, Path, os.PathLike],
            challenge: str,
            transform: Optional[Callable] = None,
            dataset_cache_file: Union[str, Path, os.PathLike] = "latent_cache.pkl",
            use_dataset_cache: bool = False
            ):
        super().__init__()

        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        
        self.dataset_cache_file = Path(dataset_cache_file)
        self.root = root
        self.challenge = challenge
        self.transform = transform
        self.raw_samples = []

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(self.root).iterdir())
            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)


                for slice_ind in range(num_slices): 
                    self.raw_samples.append(FastMRIRawDataSample(fname, slice_ind, metadata))
                
            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.raw_samples
                print(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            print(f"Using dataset cache from {self.dataset_cache_file}.")
            self.raw_samples = dataset_cache[root]

        
    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            metadata = dict(
                **hf.attrs
            )
            num_slices = hf["full_latent_tensor"].shape[0]
        return metadata, num_slices

    def __len__(self):
        return len(self.raw_samples)
    
    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.raw_samples[i]

        with h5py.File(fname, "r") as hf:
            full_latent_tensor = hf["full_latent_tensor"][dataslice]
            # masked_latent_tensor = hf["masked_latent_tensor"][dataslice]
            target = hf["target"][dataslice]
            mean_full, std_full = hf["mean_full"][dataslice], hf["std_full"][dataslice]
        
        if self.transform is not None:
            full_latent_tensor = self.transform(full_latent_tensor)
            # masked_latent_tensor = self.transform(masked_latent_tensor)
        
        return LDMSample(
            full_latent_tensor=full_latent_tensor, 
            # masked_latent_tensor=masked_latent_tensor,
            target=target,
            metadata=metadata, 
            fname=fname.name, 
            slice_num=dataslice,
            mean_full=mean_full,
            std_full=std_full)

        
class LDMLatentDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: Path,
            challenge: str, 
            batch_size: int,
            n_workers: int,
            train_transform: Optional[Callable] = None,
            val_transform: Optional[Callable] = None,
            test_transform: Optional[Callable] = None,
            use_dataset_cache_file: bool = True
            ):
        super().__init__()

        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        
        self.data_path = data_path
        self.challenge = challenge
        self.batch_size = batch_size
        self.num_workers = n_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.use_dataset_cache_file = use_dataset_cache_file


    def prepare_data(self):
        if self.use_dataset_cache_file:
            data_paths = [self.data_path / f"{self.challenge}_train", self.data_path / f"{self.challenge}_val"]
            data_transforms = [self.train_transform, self.val_transform]

            for (data_path, data_transform) in zip(data_paths, data_transforms):
                _ = LatentDataset(
                    root=data_path,
                    challenge=self.challenge,
                    transform=data_transform,
                    use_dataset_cache=self.use_dataset_cache_file
                )


    # def setup(self, stage: str):
        # if stage == "fit":
        #     self.train_dataset = LatentDataset(
        #         root=(self.data_path / f"{self.challenge}_train"),
        #         challenge=self.challenge,
        #         transform = self.train_transform,
        #         )
            
        #     self.val_dataset = LatentDataset(
        #         root=(self.data_path / f"{self.challenge}_val"),
        #         challenge=self.challenge,
        #         transform=self.val_transform
        #     )
        
        # if stage =="test":
        #     self.test_dataset = LatentDataset(
        #         root=(self.data_path / f"{self.challenge}_test"),
        #         challenge=self.challenge,
        #         transform=self.test_transform
        #     )


    def train_dataloader(self):
        self.train_dataset = LatentDataset(
            root=(self.data_path / f"{self.challenge}_train"),
            challenge=self.challenge,
            transform = self.train_transform,
            use_dataset_cache=self.use_dataset_cache_file
            )
            
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            persistent_workers=True,
            multiprocessing_context="forkserver",
            prefetch_factor=4,
            sampler=DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)
            )
    
    def val_dataloader(self):
        self.val_dataset = LatentDataset(
                root=(self.data_path / f"{self.challenge}_val"),
                challenge=self.challenge,
                transform=self.val_transform,
                use_dataset_cache=self.use_dataset_cache_file
            )
        
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            persistent_workers=True,
            multiprocessing_context="forkserver",
            prefetch_factor=4,
            sampler = DistributedSampler(self.val_dataset, shuffle=False, drop_last=True),
            )
    
    # def test_dataloader(self):
    #     self.test_dataset = LatentDataset(
    #             root=(self.data_path / f"{self.challenge}_val"),
    #             challenge=self.challenge,
    #             transform=self.test_transform
    #         )
    #     return DataLoader(
    #         self.test_dataset, 
    #         batch_size=self.batch_size, 
    #         num_workers=self.num_workers,
    #         persistent_workers=True,
    #         # multiprocessing_context="forkserver",
    #         shuffle=False)