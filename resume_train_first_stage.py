from pathlib import Path
from modules.transforms import KspaceUNetDataTransform
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from fastmri.pl_modules import FastMriDataModule
from fastmri.data.subsample import create_mask_for_mask_type
from omegaconf import OmegaConf
from modules.utils import instantiate_from_config, get_from_config
from argparse import ArgumentParser

def cli():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--id", type=str, help="WandB run id")
    return parser.parse_args()

def main(args):
    torch.set_float32_matmul_precision('high')
    config = OmegaConf.load(args.config)
    model_cfg = config.model
    data_cfg = config.data
    trainer_cfg = config.trainer
    if model_cfg.ckpt_path is None:
        model = instantiate_from_config(model_cfg)
    else: 
        model = get_from_config(model_cfg)
        model = model.load_from_checkpoint(model_cfg.ckpt_path)

    mask_func_cfg = data_cfg.mask_func
    mask_func = create_mask_for_mask_type(
        mask_func_cfg.type, 
        mask_func_cfg.center_fractions, 
        mask_func_cfg.accelerations
    )
    train_transform = KspaceUNetDataTransform(
        mask_func=mask_func, 
        adapt_pool=data_cfg.transform.adapt_pool, 
        use_seed=False
    )
    val_transform = KspaceUNetDataTransform(
        mask_func=mask_func, 
        adapt_pool=data_cfg.transform.adapt_pool
    )
    test_transform = KspaceUNetDataTransform(adapt_pool=data_cfg.transform.adapt_pool)

    dm = FastMriDataModule(
        data_path=Path(data_cfg.data_path),
        challenge=data_cfg.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        combine_train_val=False,
        test_split="test",
        test_path="/home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_test_v2",
        sample_rate=None,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.n_workers,
        distributed_sampler=False,
        use_dataset_cache_file=True
    )

    model_name = model_cfg.target.rsplit('.', 1)[-1]
    run_name = model_name + "_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    wandb.login(key="c210746318a0cf3a3fb1d542db1864e0a789e94c")
    if model_cfg.ckpt_path is None:
        wandb_logger = WandbLogger(project=model_name, name=run_name, log_model=False, config=OmegaConf.to_container(config, resolve=True))
    else: 
        wandb_logger = WandbLogger(project=model_name, name=run_name, log_model=False, config=OmegaConf.to_container(config, resolve=True), id=args.id, resume="must")

    ckpt_config = trainer_cfg.checkpoint
    model_checkpoint = ModelCheckpoint(
        save_top_k=ckpt_config.top_k,
        monitor=ckpt_config.monitor,
        mode=ckpt_config.mode,
        filename=model_name + "-{epoch:02d}"
    )

    trainer = pl.Trainer(max_epochs=trainer_cfg.max_epochs, logger=wandb_logger, callbacks=[model_checkpoint])
    trainer.fit(model, dm)
    wandb.finish()


if __name__ == "__main__":
    args = cli()
    main(args)
    