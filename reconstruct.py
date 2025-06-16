from pathlib import Path
from modules.transforms import KspaceUNetDataTransform, KspaceUNetDataTransform320
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_modules.ae_unet_module import Kspace_AE_Unet, Kspace_AE_Unet_SSIM
import torch
from fastmri.pl_modules import FastMriDataModule
from fastmri.data.subsample import create_mask_for_mask_type

torch.set_float32_matmul_precision('high')
if __name__ == "__main__":
    config = {
        "mask_type": "equispaced_fraction",
        "center_fractions": [0.04],
        "accelerations": [8],
        "cropped": False, 
        "batch_size": 1,
        "domain": "Kspace",
        "loss": "L1", 
        "latent_dim": 128,
        "n_channels": 32,
        "epochs": 100
    }

    assert config["domain"] in ("Kspace", "CImage"), "You can only select 'Kspace' or 'CImage' as domain"
    assert config["loss"] in ("L1", "SSIM"), "You can only select 'L1' or 'SSIM' as objective function"
    # assert (config["batch_size"] > 1) and (config["cropped"]), "When the 'batch_size' is > 1, then you have to set 'cropped' to TRUE"

    model_name = config["domain"] + "_AE_Unet"

    if config["loss"] == "SSIM":
        model_name += "_SSIM"

    mask_func = create_mask_for_mask_type(
        config["mask_type"], config["center_fractions"], config["accelerations"]
    )
    train_transform = KspaceUNetDataTransform(mask_func=mask_func, use_seed=False)
    val_transform = KspaceUNetDataTransform(mask_func=mask_func)
    test_transform = KspaceUNetDataTransform()

    if config["cropped"]:
        model_name += "_320"
        train_transform = KspaceUNetDataTransform320(mask_func=mask_func, use_seed=False)
        val_transform = KspaceUNetDataTransform320(mask_func=mask_func)
        test_transform = KspaceUNetDataTransform320()

    print(model_name)
    
    if config["loss"] == "SSIM":
        model = Kspace_AE_Unet_SSIM(n_channels=config["n_channels"], latent_dim=config["latent_dim"])
    else:
        model = Kspace_AE_Unet(n_channels=config["n_channels"], latent_dim=config["latent_dim"])

    
    run_name = model_name + "_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    wandb.login(key="c210746318a0cf3a3fb1d542db1864e0a789e94c")
    wandb_logger = WandbLogger(project=model_name[:14], name=run_name, log_model=False, config=config)

    model_checkpoint = ModelCheckpoint(
        save_top_k=2,
        monitor="val/ssim_epoch",
        mode="min",
        filename=model_name[:14] + "-{epoch:02d}"
    )

    trainer = pl.Trainer(max_epochs=config["epochs"], logger=wandb_logger, callbacks=[model_checkpoint])

    data_module = FastMriDataModule(
        data_path=Path("/home/saturn/iwai/iwai113h/IdeaLab/knee_dataset"),
        challenge="singlecoil",
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        combine_train_val=False,
        test_split="test",
        sample_rate=None,
        batch_size=config["batch_size"],
        num_workers=4,
        distributed_sampler=False,
        use_dataset_cache_file=True
    )
    trainer.fit(model,datamodule=data_module)
    wandb.finish()