import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import os
from typing import Union, Optional, Callable
import h5py
from fastmri.data.mri_data import FastMRIRawDataSample
from modules.transforms import LDMSample

class LatentDataset(Dataset):
    def __init__(
            self,
            root: Union[str, Path, os.PathLike],
            challenge: str,
            transform: Optional[Callable] = None
            ):
        super().__init__()

        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        
        self.root = root
        self.challenge = challenge
        self.transform = transform

        files = list(Path(self.root).iterdir())
        self.raw_samples = []
        for fname in sorted(files):
            metadata, num_slices = self._retrieve_metadata(fname)

            for slice_ind in range(num_slices):
                self.raw_samples.append(FastMRIRawDataSample(fname, slice_ind, metadata))

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
            masked_latent_tensor = hf["masked_latent_tensor"][dataslice]
            mean_full, std_full = hf["mean_full"][dataslice], hf["std_full"][dataslice]
        
        if self.transform is not None:
            full_latent_tensor = self.transform(full_latent_tensor)
            masked_latent_tensor = self.transform(masked_latent_tensor)
        
        return LDMSample(
            full_latent_tensor=full_latent_tensor, 
            masked_latent_tensor=masked_latent_tensor,
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
            num_workers: int,
            train_transform: Optional[Callable] = None,
            val_transform: Optional[Callable] = None,
            test_transform: Optional[Callable] = None
            ):
        super().__init__()

        self.data_path = data_path
        self.challenge = challenge
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform


    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = LatentDataset(
                root=(self.data_path / f"{self.challenge}_train"),
                challenge=self.challenge,
                transform = self.train_transform,
                )
            
            self.val_dataset = LatentDataset(
                root=(self.data_path / f"{self.challenge}_val"),
                challenge=self.challenge,
                transform=self.val_transform
            )
        
        if stage =="test":
            self.test_dataset = LatentDataset(
                root=(self.data_path / f"{self.challenge}_test"),
                challenge=self.challenge,
                transform=self.test_transform
            )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False)