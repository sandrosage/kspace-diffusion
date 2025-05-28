from fastmri.pl_modules import FastMriDataModule
import torch
from fastmri.data.subsample import create_mask_for_mask_type
from pathlib import Path
from modules.transforms import KspaceUNetDataTransform
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_modules.myUnet_module import MyUnetModule
from torch.nn import L1Loss
# export https_proxy=http://proxy:80
# /home/saturn/iwai/iwai113h/IdeaLab/knee_dataset
# /vol/datasets/cil/2021_11_23_fastMRI_data/knee/unzipped

torch.set_float32_matmul_precision('high')
if __name__ == "__main__":
    config = {
        "mask_type": "equispaced_fraction",
        "center_fractions": [0.04],
        "accelerations": [8],
        "loss_domain": "ssim", 
        "criterion": L1Loss(),
        "n_channels": 128,
        "with_residual": False,
        "latent_dim": 128,
        "mode": "interpolation"
    }

    if config["with_residual"]: 
        model_name = "Unet_"
    else:
        model_name = "NoSkipUnet_"
    model_name = model_name + config["mode"] + "_"
    run_name = model_name + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    wandb.login(key="c210746318a0cf3a3fb1d542db1864e0a789e94c")
    wandb_logger = WandbLogger(project="Kspace-Experiments", name=run_name, log_model=True, config=config)

    model = MyUnetModule(loss_domain=config["loss_domain"], n_channels=config["n_channels"], num_log_images=32, with_residual=config["with_residual"], latent_dim=config["latent_dim"], mode=config["mode"])

    model_checkpoint = ModelCheckpoint(
        save_top_k=2,
        monitor="val/ssim_loss_epoch",
        mode="min",
        filename="myunet-{epoch:02d}"
    )
    trainer = pl.Trainer(max_epochs=2, logger=wandb_logger, callbacks=[model_checkpoint])

    mask_func = create_mask_for_mask_type(
        config["mask_type"], config["center_fractions"], config["accelerations"]
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = KspaceUNetDataTransform(mask_func=mask_func, use_seed=False)
    val_transform = KspaceUNetDataTransform(mask_func=mask_func)
    test_transform = KspaceUNetDataTransform()
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
    trainer.fit(model,datamodule=data_module)
    wandb.finish()
