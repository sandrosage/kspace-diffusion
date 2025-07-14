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
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to latent dataset")
    args = parser.parse_args()
    config = {
        "epochs": 100,
        "in_channels": 16,
        "out_channels": 16,
        "batch_size": 32
    }

    torch.set_float32_matmul_precision('high')

    model_name = "LDM"

    print(model_name)
    
    model = LDM(in_channels=config["in_channels"], out_channels=config["out_channels"])

    
    run_name = model_name + "_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    wandb.login(key="c210746318a0cf3a3fb1d542db1864e0a789e94c")
    wandb_logger = WandbLogger(project=model_name[:4], name=run_name, log_model=False, config=config)

    model_checkpoint = ModelCheckpoint(
        save_top_k=2,
        monitor="val/mse_noise_loss_epoch",
        mode="min",
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