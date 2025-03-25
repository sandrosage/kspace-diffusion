from fastmri.pl_modules import FastMriDataModule, MriModule
import torch
from fastmri.data.subsample import create_mask_for_mask_type
from pathlib import Path
from modules.transforms import KspaceLDMDataTransform, Kspace3DLDMDataTransform
from modules.autoencoders import Encoder,Decoder
from pl_modules.new import AutoencoderKL
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

model_checkpoint = ModelCheckpoint(
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        filename="pixelcnn-kspace-{epoch:02d}-{val_loss:.2f}"
    )

if __name__ == "__main__":
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
    trainer = pl.Trainer(devices=1, max_epochs=150)
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
    dm = FastMriDataModule(
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

    trainer.fit(model,datamodule=dm)