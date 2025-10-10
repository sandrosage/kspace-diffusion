from pl_modules.diffusers_vae_module import KspaceAutoencoder, KspaceAutoencoderKL
from modules.transforms import KspaceUNetDataTransform, norm
from pathlib import Path
from fastmri.data import SliceDataset
import h5py
from fastmri.data.subsample import create_mask_for_mask_type
import os
import torch
from collections import Counter

partition = "train"
challenge = "singlecoil"
config = {
        "mask_type": "equispaced_fraction",
        "center_fractions": [0.08],
        "accelerations": [4],
    }
mask_func = create_mask_for_mask_type(
    config["mask_type"], config["center_fractions"], config["accelerations"]
)

train_transform = KspaceUNetDataTransform(mask_func=mask_func, use_seed=False, adapt_pool=True)
ds = SliceDataset(
    root=Path("/home/janus/iwbi-cip-datasets/shared/fastMRI/knee/" + f"{challenge}_{partition}/"),
    transform=train_transform,
    challenge=challenge,
    use_dataset_cache=True
)

dl = torch.utils.data.DataLoader(ds)
w, h = [], []
for i, batch in enumerate(dl):
    kspace = batch.full_kspace.permute(0,3,1,2).contiguous()
    new_w, new_h = kspace.shape[-2], kspace.shape[-1]
    w.append(new_w)
    h.append(new_h)
    if i % 100 == 0:
        print(i, new_w, new_h)
print(f"Max kspace size in {partition} set: {w} x {h}")
print(Counter(w), Counter(h))