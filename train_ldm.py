from pathlib import Path
from pl_modules.data_module import LDMLatentDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pl_modules.ldm_module import LDM
from argparse import ArgumentParser
from pl_modules.diffusers_vae_module import KspaceAutoencoder, KspaceAutoencoderKL
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to latent dataset")
    args = parser.parse_args()
    config = {
        "epochs": 100,
        "ckpt_path": "KspaceAutoencoder-epoch=11.ckpt",
        # "model_ckpt_path": "LDM_/88aswu02/checkpoints/LDM_-epoch=17.ckpt"
        "model_ckpt_path": None
    }

    torch.set_float32_matmul_precision('high')

    model_name = "LDM_first_real_try"

    print(model_name)
    model_type = "KspaceAutoencoder"
    id = "i6nac0hp"
    ckpt_path = config["ckpt_path"]
    first_stage = KspaceAutoencoder.load_from_checkpoint(f"{model_type}/{id}/checkpoints/{ckpt_path}")

    if config["model_ckpt_path"] is not None:
        print("Use model checkpoint...")
        model = LDM.load_from_checkpoint(config["model_ckpt_path"], first_stage=first_stage)
    else:
        model = LDM(in_channels=config["in_channels"], out_channels=config["out_channels"], first_stage=first_stage)

    
    run_name = model_name + "_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    wandb.login(key="c210746318a0cf3a3fb1d542db1864e0a789e94c")
    wandb_logger = WandbLogger(project=model_name[:4], name=run_name, log_model=False, config=config)

    model_checkpoint = ModelCheckpoint(
        save_top_k=2,
        monitor="val/mse_noise_loss_epoch",
        mode="min",
        save_last=True,
        filename=model_name[:4] + "-{epoch:02d}"
    )

    dm = LDMLatentDataModule(
    data_path=Path(args.path),
    challenge="singlecoil",
    batch_size=config["batch_size"],
    num_workers=4, 
)
    
    trainer = pl.Trainer(max_epochs=config["epochs"], logger=wandb_logger, callbacks=[model_checkpoint])

    trainer.fit(model,datamodule=dm)
    wandb.finish()