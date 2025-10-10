import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf
from modules.utils import instantiate_from_config, get_from_config
from pathlib import Path

def cli():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--id", type=str, default=None, help="WandB run id")
    return parser.parse_args()

def main(args):
    torch.set_float32_matmul_precision('high')
    config = OmegaConf.load(args.config)
    model_cfg = config.model
    data_cfg = config.data
    trainer_cfg = config.trainer
    fs_cfg = config.first_stage

    first_stage = get_from_config(fs_cfg)
    first_stage = first_stage.load_from_checkpoint(fs_cfg.ckpt_path)


    model = get_from_config(model_cfg)
    if model_cfg.ckpt_path is not None:
        print(f"Use Checkpoint: {model_cfg.ckpt_path}")
        model = model.load_from_checkpoint(model_cfg.ckpt_path, first_stage=first_stage)
    else:
        model = model(in_channels=model_cfg.params.in_channels, out_channels = model_cfg.params.out_channels, first_stage=first_stage)

    if not isinstance(data_cfg.params.data_path, Path):
        data_cfg.params.data_path = Path(data_cfg.params.data_path) 
    dm = instantiate_from_config(data_cfg)

    model_name = "LDM_" + model_cfg.target.rsplit('.', 1)[-1]
    run_name = model_name + "_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    wandb.login(key="c210746318a0cf3a3fb1d542db1864e0a789e94c")

    if args.id is not None:
        wandb_logger = WandbLogger(project=model_name[:4], name=run_name, log_model=False, config=OmegaConf.to_container(config, resolve=True), id=args.id, resume="must")
    else:
        wandb_logger = WandbLogger(project=model_name[:4], name=run_name, log_model=False, config=OmegaConf.to_container(config, resolve=True))

    ckpt_config = trainer_cfg.checkpoint
    model_checkpoint = ModelCheckpoint(
        save_top_k=ckpt_config.top_k,
        monitor=ckpt_config.monitor,
        mode=ckpt_config.mode,
        save_last=True,
        filename=model_name + "-{epoch:02d}"
    )

    trainer = pl.Trainer(max_epochs=trainer_cfg.max_epochs, logger=wandb_logger, callbacks=[model_checkpoint])

    # trainer = pl.Trainer(max_epochs=trainer_cfg.max_epochs, logger=wandb_logger, callbacks=[model_checkpoint], strategy="ddp_find_unused_parameters_true")
    if model_cfg.ckpt_path is not None:
        print(f"Trainer using Checkpoint path: {model_cfg.ckpt_path}")
        trainer.fit(model, dm, ckpt_path=model_cfg.ckpt_path)
    else:
        trainer.fit(model, dm)
    wandb.finish()

if __name__ == "__main__":
   args = cli()
   main(args)