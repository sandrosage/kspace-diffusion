from pl_modules.unet_module import UNet
from modules.transforms import KspaceUNetDataTransform
from fastmri.pl_modules import FastMriDataModule
from pathlib import Path
from modules.transforms import kspace_to_mri
import matplotlib.pyplot as plt
from fastmri.data.subsample import create_mask_for_mask_type
import torch

def empty_residuals(residuals: list):
    return [torch.zeros_like(r).to("cuda") for r in residuals]

mask_type = "random"
center_fractions = [0.04]
accelerations = [4]
mask = create_mask_for_mask_type(
    mask_type, center_fractions, accelerations
)

train_transform = val_transform = test_transform = KspaceUNetDataTransform(mask_func=mask)
# train_transform = val_transform = test_transform = UnetDataTransform(which_challenge="singlecoil")
# ptl data module - this handles data loaders
data_module = FastMriDataModule(
    data_path=Path("/home/janus/iwbi-cip-datasets/shared/fastMRI/knee"),
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
    use_dataset_cache_file=True
)
model = UNet.load_from_checkpoint("UNet/hdzt9rut/checkpoints/UNet-epoch=09.ckpt", strict=False)
model.eval()

for i, batch in enumerate(data_module.val_dataloader()):
    with torch.no_grad():
        if batch.fname[0] != "file1001977.h5":
            continue
        if batch.slice_num[0] != 16:
            continue
        print("Found")
        input =  batch.full_kspace
        input = input.permute(0,3,1,2).contiguous().to("cuda")
        lst, residuals = model.downsample(input)
        output = model.upsample(lst, residuals)
        output = output.permute(0,2,3,1).contiguous()
        output_full = kspace_to_mri(output)
        lst_0, residuals_0 = model.downsample(input)
        output_only_lst = kspace_to_mri(model.upsample(lst_0, empty_residuals(residuals_0)).permute(0,2,3,1).contiguous())
        lst_1, residuals_1 = model.downsample(input)
        output_only_residual = kspace_to_mri(model.upsample(torch.zeros_like(lst_1).to("cuda"), residuals_1).permute(0,2,3,1).contiguous())
        plt.imshow(output_full.squeeze(0).cpu().numpy(), cmap='gray')
        plt.tight_layout()
        plt.axis("off")
        plt.savefig("unet_reconstruction.pdf", format="pdf", bbox_inches='tight', pad_inches = 0, dpi=1000)
        plt.close()
        plt.imshow(output_only_lst.squeeze(0).cpu().numpy(), cmap='gray')
        plt.tight_layout()
        plt.axis("off")
        plt.savefig("unet_lst.pdf", format="pdf", bbox_inches='tight', pad_inches = 0, dpi=1000)
        plt.close()
        plt.imshow(output_only_residual.squeeze(0).cpu(), cmap='gray')
        plt.tight_layout()
        plt.axis("off")
        plt.savefig("unet_residual.pdf", format="pdf", bbox_inches='tight', pad_inches = 0, dpi=1000)
        plt.close()
        break
