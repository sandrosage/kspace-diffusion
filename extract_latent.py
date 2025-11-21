from pl_modules.diffusers_vae_module import WeightedSSIMKspaceAutoencoderKL, WeightedSSIMKspaceAutoencoder, KspaceAutoencoder, KspaceAutoencoderKL
from modules.transforms import KspaceUNetDataTransform, norm
from pathlib import Path
from fastmri.data import SliceDataset
import h5py
from fastmri.data.subsample import create_mask_for_mask_type
import os
import torch
from argparse import ArgumentParser

def extract_between_first_two_slashes(path: str) -> str:
    """
    Extracts the substring between the first and second '/' in a given string.
    
    Example:
        "UNet/77iwp3bt/checkpoints/epoch=99-step=6000.ckpt"
    """
    parts = path.split('/')
    if len(parts) < 3:
        raise ValueError("Input string must contain at least two '/' characters")
    return parts[0], parts[1]

def cli():
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str, help="Path to ckpt file")
    parser.add_argument("--partition", type=str, default="train", help="Dataset partition: train/val/test")
    return parser.parse_args()

def main(args):
    ckpt_path = args.ckpt
    model_type, id = extract_between_first_two_slashes(ckpt_path)
    
    if model_type == "WeightedSSIMKspaceAutoencoderKL":
        model = WeightedSSIMKspaceAutoencoderKL.load_from_checkpoint(ckpt_path).eval()
    elif model_type == "WeightedSSIMKspaceAutoencoder":
        model = WeightedSSIMKspaceAutoencoder.load_from_checkpoint(ckpt_path, strict=False).eval()   
    elif model_type == "KspaceAutoencoder":
        model = KspaceAutoencoder.load_from_checkpoint(ckpt_path, strict=False)
    elif model_type == "KspaceAutoencoderKL":
        model = KspaceAutoencoderKL.load_from_checkpoint(ckpt_path, strict=False)
    else:
        raise ValueError(f"Model type {model_type} not supported")
    
    print("Model loaded:", model_type, id)
    for p in model.parameters():
        p.requires_grad = False
    
    down_factor = 2**(model.down_layers-1)
    print("Downsampling factor: ", down_factor)
    out_shape = (0, model.latent_dim, 640 // down_factor, 384 // down_factor)
    print(out_shape)
        
    partition = args.partition  # "train", "val", "test"
    challenge = "singlecoil"
    path = f"/home/atuin/b180dc/b180dc46/{model_type}/{id}/" + f"{challenge}_{partition}/"
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

    model.eval()
    print(f"Starting latent extraction of {model_type}_{id}...")
    dl = torch.utils.data.DataLoader(ds)
    current_fname = next(iter(dl)).fname[0]
    hf = h5py.File(path + current_fname, "w")
    full_ds = hf.create_dataset("full_latent_tensor", shape=out_shape, maxshape=(None, 16, 160, 96), dtype="float32")
    masked_ds = hf.create_dataset("masked_latent_tensor", shape=out_shape, maxshape=(None, 16, 160, 96), dtype="float32")
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
            full_ds = hf.create_dataset("full_latent_tensor", shape=out_shape, maxshape= (None, 16, 160, 96), dtype="float32")
            masked_ds = hf.create_dataset("masked_latent_tensor", shape=out_shape, maxshape= (None, 16, 160, 96), dtype="float32")
            target_ds = hf.create_dataset("target", shape=(0, 1, 320, 320), maxshape=(None, 1, 320, 320), dtype="float32")
            mean_full_ds = hf.create_dataset("mean_full", shape=(0, 2, 1, 1), maxshape=(None, 2, 1, 1), dtype="float32")
            std_full_ds = hf.create_dataset("std_full", shape=(0, 2, 1, 1), maxshape=(None, 2, 1, 1), dtype="float32")
            mean_masked_ds = hf.create_dataset("mean_masked", shape=(0, 2, 1, 1), maxshape=(None, 2, 1, 1), dtype="float32")
            std_masked_ds = hf.create_dataset("std_masked", shape=(0, 2, 1, 1), maxshape=(None, 2, 1, 1), dtype="float32")
            current_fname = fname

        full_kspace = batch.full_kspace.permute(0,3,1,2).contiguous()
        full_kspace, mean_full, std_full = norm(full_kspace)
        with torch.no_grad():
            full_latent_tensor = model.encode(full_kspace.cuda())[0].cpu()
        if full_latent_tensor.ndim > 3:
            full_latent_tensor = full_latent_tensor.squeeze(0)
        full_ds.resize((slice_num + 1, *full_latent_tensor.shape))
        full_ds[slice_num] = full_latent_tensor.detach().numpy()
        mean_full_ds.resize((slice_num + 1, 2, 1, 1))
        mean_full_ds[slice_num] = mean_full
        std_full_ds.resize((slice_num + 1, 2, 1, 1))
        std_full_ds[slice_num] = std_full

        masked_kspace = batch.masked_kspace.permute(0,3,1,2).contiguous()
        masked_kspace, mean_masked, std_masked = norm(masked_kspace)
        with torch.no_grad():
            masked_latent_tensor = model.encode(masked_kspace.cuda())[0].cpu()
        if masked_latent_tensor.ndim > 3:
            masked_latent_tensor = masked_latent_tensor.squeeze(0)
        masked_ds.resize((slice_num + 1, *masked_latent_tensor.shape))
        masked_ds[slice_num] = masked_latent_tensor.detach().numpy()
        mean_masked_ds.resize((slice_num + 1, 2, 1, 1))
        mean_masked_ds[slice_num] = mean_masked
        std_masked_ds.resize((slice_num + 1, 2, 1, 1))
        std_masked_ds[slice_num] = std_masked
        target_ds.resize((slice_num + 1, *batch.target.shape))
        target_ds[slice_num] = batch.target
        hf.attrs["num_low_frequencies"] = num_low_frequencies

if __name__ == "__main__":
    args = cli()
    main(args)
