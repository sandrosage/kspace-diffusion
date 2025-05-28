from pl_modules.myUnet_module import MyUnetModule
from modules.transforms import KspaceUNetDataTransform
from pathlib import Path
from fastmri.pl_modules import FastMriDataModule
import h5py
from fastmri.data.subsample import create_mask_for_mask_type
from torch.nn import L1Loss
import torch

path = "latent_data/"
config = {
        "mask_type": "equispaced_fraction",
        "center_fractions": [0.04],
        "accelerations": [8],
        "with_dc": True,
        "loss_domain": "ssim", 
        "n_channels": 128,
        "soft_dc": False,
        "criterion": L1Loss()
    }
mask_func = create_mask_for_mask_type(
    config["mask_type"], config["center_fractions"], config["accelerations"]
)

train_transform = KspaceUNetDataTransform(mask_func=mask_func, use_seed=False)
val_transform = KspaceUNetDataTransform(mask_func=mask_func)
test_transform = KspaceUNetDataTransform()
# ptl data module - this handles data loaders
data_module = FastMriDataModule(
    data_path=Path("/vol/datasets/cil/2021_11_23_fastMRI_data/knee/unzipped"),
    challenge="singlecoil",
    train_transform=train_transform,
    val_transform=val_transform,
    test_transform=test_transform,
    combine_train_val=False,
    test_split="test",
    sample_rate=None,
    batch_size=1,
    num_workers=4,
    distributed_sampler=False,
    use_dataset_cache_file=False
)
model = MyUnetModule.load_from_checkpoint("checkpoints/epoch=7-step=277936.ckpt", criterion=config["criterion"])
train_dl = data_module.train_dataloader()

for i, batch in enumerate(train_dl):
    # fname = batch.fname[0]
    # slice_num = batch.slice_num
    # full_kspace = batch.full_kspace.permute(0,3,1,2).contiguous()
    # masked_kspace = batch.masked_kspace.permute(0,3,1,2).contiguous()
    # full_latent_tensor = model.downsample(full_kspace.cuda())[0].cpu()
    # masked_latent_tensor = model.downsample(masked_kspace.cuda())[0].cpu()
    # with h5py.File("latent_data/" + fname[:-3] + "_" + str(slice_num.numpy()[0]) + ".h5", "w") as hf:
    #     hf.create_dataset("full_latent_tensor", data=full_latent_tensor.detach().numpy())
    #     hf.create_dataset("masked_latent_tensor", data=masked_latent_tensor.detach().numpy())
    #     hf.attrs["num_low_frequencies"] = batch.num_low_frequencies
    #     hf.close()
    fname = batch.fname
    if i == 0:
        file_name = fname
        masked_latent_tensors_list = []
        full_latent_tensors_list = []
    else:
        if fname != file_name:
            print(fname, file_name)
            full_combined_slices = torch.stack(full_latent_tensors_list, dim=0)
            masked_combined_slices = torch.stack(masked_latent_tensors_list, dim=0)
            with h5py.File(path + fname, "w") as hf:
                hf.create_dataset("full_latent_tensor", data=full_combined_slices)
                hf.create_dataset("subsampled_latent_tensor", data=masked_combined_slices)
                hf.attrs["num_low_frequencies"] = num_low_frequencies
            file_name = fname
            full_latent_tensors_list = []
            masked_latent_tensors_list = []
    num_low_frequencies = batch.num_low_frequencies
    full_kspace = batch.full_kspace.permute(0,3,1,2).contiguous()
    masked_kspace = batch.masked_kspace.permute(0,3,1,2).contiguous()
    full_latent_tensor = model.downsample(full_kspace.cuda())[0]
    print(full_latent_tensor.shape)
    del full_kspace
    masked_latent_tensor = model.downsample(masked_kspace.cuda())[0]
    del masked_kspace
    full_latent_tensors_list.append(full_latent_tensor)
    masked_latent_tensors_list.append(masked_latent_tensor)
