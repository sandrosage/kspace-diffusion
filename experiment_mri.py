from fastmri.pl_modules import FastMriDataModule
import torch
from fastmri.data.subsample import create_mask_for_mask_type
from pathlib import Path
from modules.transforms import KspaceLDMDataTransform, LogPhaseLDMDataTransform
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pl_modules.autoencoder_module import AutoencoderKL, AutoencoderComplex
from pl_modules.rupali_autoencoder import RupaliAutoencoderModule
from pl_modules.simple_module import SimpleAutoencoder
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint

# export https_proxy=http://proxy:80

torch.set_float32_matmul_precision('high')
if __name__ == "__main__":
    print("Hello")
    run_name = "Simple-Euler-OnlyKspace-Loss_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    wandb.login(key="c210746318a0cf3a3fb1d542db1864e0a789e94c")
    wandb_logger = WandbLogger(project="Kspace-Diffusion", name=run_name, log_model=True)
    dd_config = {
      "double_z": True,
      "z_channels": 1,
      "resolution": 256,
      "in_channels": 1,
      "out_ch": 1,
      "ch": 128,
      "ch_mult": [1,2,4],  # num_down = len(ch_mult)-1
      "num_res_blocks": 1,
      "attn_resolutions": [],
      "dropout": 0.0
    }
    # model = AutoencoderComplex(ddconfig=dd_config, lossconfig=None, embed_dim=1)
    # model = RupaliAutoencoderModule()
    model = SimpleAutoencoder()

    model_checkpoint = ModelCheckpoint(
        save_top_k=2,
        monitor="val/ssim_loss_epoch",
        mode="min",
        filename="autoencoder-kl-{epoch:02d}"
    )
    trainer = pl.Trainer(devices=1, max_epochs=150, logger=wandb_logger, callbacks=[model_checkpoint])
    mask_type = "random"
    center_fractions = [0.08, 0.04]
    accelerations = [4, 8]
    mask = create_mask_for_mask_type(
        mask_type, center_fractions, accelerations
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = val_transform = test_transform = LogPhaseLDMDataTransform()
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=Path("/home/saturn/iwai/iwai113h/IdeaLab/knee_dataset"),
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
    trainer.fit(model,train_dataloaders=data_module.train_dataloader())
    wandb.finish()
