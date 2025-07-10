from pl_modules.diffusers_vae_module import Diffusers_VAE
from modules.transforms import KspaceUNetDataTransform, norm
from pathlib import Path
from fastmri.data import SliceDataset
import h5py
from fastmri.data.subsample import create_mask_for_mask_type
import os
import torch

partition = "test_v2"
challenge = "singlecoil"
path = "/home/atuin/b180dc/b180dc46/Diffusers_VAE_16_4/latent_data/" + f"{challenge}_{partition}/"
if not os.path.exists(path):
    os.makedirs(path)
config = {
        "mask_type": "equispaced_fraction",
        "center_fractions": [0.08],
        "accelerations": [4],
    }
mask_func = create_mask_for_mask_type(
    config["mask_type"], config["center_fractions"], config["accelerations"]
)

train_transform = KspaceUNetDataTransform(mask_func=mask_func, use_seed=False)
ds = SliceDataset(
    root=Path("/home/janus/iwbi-cip-datasets/shared/fastMRI/knee/" + f"{challenge}_{partition}/"),
    transform=train_transform,
    challenge=challenge,
    use_dataset_cache=True
)

model = Diffusers_VAE.load_from_checkpoint("Diffusers_VAE/u80szjw0/checkpoints/Diffusers_VAE_-epoch=12.ckpt", down_layers=4)

print("Starting...")
dl = torch.utils.data.DataLoader(ds)
current_fname = next(iter(dl)).fname[0]
hf = h5py.File(path + current_fname, "w")
full_ds = hf.create_dataset("full_latent_tensor", shape=(0, 16, 80, 46), maxshape=(None, 16, 80, 100), dtype="float32")
masked_ds = hf.create_dataset("masked_latent_tensor", shape=(0, 16, 80, 46), maxshape=(None, 16, 80, 100), dtype="float32")
target_ds = hf.create_dataset("target", shape=(0, 1, 320, 320), maxshape=(None, 1, 320, 320), dtype="float32")
mean_full_ds = hf.create_dataset("mean_full", shape=(0, 2, 1, 1), maxshape=(None, 2, 1, 1), dtype="float32")
std_full_ds = hf.create_dataset("std_full", shape=(0, 2, 1, 1), maxshape=(None, 2, 1, 1), dtype="float32")
mean_masked_ds = hf.create_dataset("mean_masked", shape=(0, 2, 1, 1), maxshape=(None, 2, 1, 1), dtype="float32")
std_masked_ds = hf.create_dataset("std_masked", shape=(0, 2, 1, 1), maxshape=(None, 2, 1, 1), dtype="float32")
for i, batch in enumerate(dl):
    fname = batch.fname[0]
    slice_num = batch.slice_num.item()
    num_low_frequencies =  batch.num_low_frequencies.item()
    # print(fname, slice_num, num_low_frequencies)
    if fname != current_fname:
        print("Done: ", current_fname)
        hf.close()
        hf = h5py.File(path + fname, "w")
        full_ds = hf.create_dataset("full_latent_tensor", shape=(0, 16, 80, 46), maxshape= (None, 16, 80, 100), dtype="float32")
        masked_ds = hf.create_dataset("masked_latent_tensor", shape=(0, 16, 80, 46), maxshape= (None, 16, 80, 100), dtype="float32")
        target_ds = hf.create_dataset("target", shape=(0, 1, 320, 320), maxshape=(None, 1, 320, 320), dtype="float32")
        mean_full_ds = hf.create_dataset("mean_full", shape=(0, 2, 1, 1), maxshape=(None, 2, 1, 1), dtype="float32")
        std_full_ds = hf.create_dataset("std_full", shape=(0, 2, 1, 1), maxshape=(None, 2, 1, 1), dtype="float32")
        mean_masked_ds = hf.create_dataset("mean_masked", shape=(0, 2, 1, 1), maxshape=(None, 2, 1, 1), dtype="float32")
        std_masked_ds = hf.create_dataset("std_masked", shape=(0, 2, 1, 1), maxshape=(None, 2, 1, 1), dtype="float32")
        current_fname = fname

    full_kspace = batch.full_kspace.permute(0,3,1,2).contiguous()
    full_kspace, mean_full, std_full = norm(full_kspace)
    full_latent_tensor = model.encode(full_kspace.cuda())[0].cpu()
    full_ds.resize((slice_num + 1, *full_latent_tensor.shape))
    full_ds[slice_num] = full_latent_tensor.detach().numpy()
    mean_full_ds.resize((slice_num + 1, 2, 1, 1))
    mean_full_ds[slice_num] = mean_full
    std_full_ds.resize((slice_num + 1, 2, 1, 1))
    std_full_ds[slice_num] = std_full

    masked_kspace = batch.masked_kspace.permute(0,3,1,2).contiguous()
    masked_kspace, mean_masked, std_masked = norm(masked_kspace)
    masked_latent_tensor = model.encode(masked_kspace.cuda())[0].cpu()
    masked_ds.resize((slice_num + 1, *masked_latent_tensor.shape))
    masked_ds[slice_num] = masked_latent_tensor.detach().numpy()
    mean_masked_ds.resize((slice_num + 1, 2, 1, 1))
    mean_masked_ds[slice_num] = mean_masked
    std_masked_ds.resize((slice_num + 1, 2, 1, 1))
    std_masked_ds[slice_num] = std_masked
    target_ds.resize((slice_num + 1, *batch.target.shape))
    target_ds[slice_num] = batch.target
    hf.attrs["num_low_frequencies"] = num_low_frequencies
