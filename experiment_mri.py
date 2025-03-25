from fastmri.pl_modules import FastMriDataModule
import torch
from fastmri.data.subsample import create_mask_for_mask_type
from pathlib import Path
from modules.transforms import KspaceLDMDataTransform
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pl_modules import AutoencoderKL
torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    wandb.login(key="c210746318a0cf3a3fb1d542db1864e0a789e94c")
    wandb_logger = WandbLogger(project="Kspace-Diffusion", name="First-Stage-AutoencoderKL", log_model=True)
    dd_config = {
      "double_z": True,
      "z_channels": 2,
      "resolution": 256,
      "in_channels": 2,
      "out_ch": 2,
      "ch": 128,
      "ch_mult": [1,2,4],  # num_down = len(ch_mult)-1
      "num_res_blocks": 2,
      "attn_resolutions": [],
      "dropout": 0.0
    }
    model = AutoencoderKL(ddconfig=dd_config, lossconfig=None, embed_dim=2)
    trainer = pl.Trainer(devices=1, max_epochs=150, logger=wandb_logger)
    mask_type = "random"
    center_fractions = [0.08, 0.04]
    accelerations = [4, 8]
    mask = create_mask_for_mask_type(
        mask_type, center_fractions, accelerations
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = KspaceLDMDataTransform()
    val_transform = KspaceLDMDataTransform()
    test_transform = KspaceLDMDataTransform()
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
        use_dataset_cache_file=True
    )
    trainer.fit(model,train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
    wandb.finish()