from pathlib import Path
from modules.transforms import KspaceUNetDataTransform
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from datetime import datetime
import torch
from fastmri.pl_modules import FastMriDataModule
from fastmri.data.subsample import create_mask_for_mask_type
from omegaconf import OmegaConf
from modules.utils import get_from_config
from argparse import ArgumentParser
import json

def extract_between_first_two_slashes(path: str) -> str:
    """
    Extracts the substring between the first and second '/' in a given string.
    
    Example:
        "UNet/77iwp3bt/checkpoints/epoch=99-step=6000.ckpt" -> "77iwp3bt"
    """
    parts = path.split('/')
    if len(parts) < 3:
        raise ValueError("Input string must contain at least two '/' characters")
    return parts[0],parts[1]


def cli():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--undersampling", action="store_true", help="Flag for evaluation on undersampled k-space")
    parser.add_argument("--accelerations", type = int, choices=[4,8], default= 4, help="Acceleration factor for undersampling mask")
    parser.add_argument("--mask_type", type=str, choices=["random", "equispaced"], default="equispaced", help="Mask type for undersampling")
    return parser.parse_args()

def main(args):
    torch.set_float32_matmul_precision('high')
    config = OmegaConf.load(args.config)
    model_cfg = config.model
    data_cfg = config.data
    try:
        model_name, model_cfg.id = extract_between_first_two_slashes(model_cfg.ckpt_path)
        print("Extracted ID:", model_cfg.id)
    except ValueError as e:
        print("Error extracting ID:", e)
        model_cfg.id = None

    assert model_cfg.ckpt_path is not None, "No checkpoint path provided in the config file."

    print(f"Use Checkpoint: {model_cfg.ckpt_path}")
    model = get_from_config(model_cfg)
    model = model.load_from_checkpoint(model_cfg.ckpt_path, strict=False, undersampling=args.undersampling)

    mask_func = create_mask_for_mask_type(
        args.mask_type, 
        [0.08], 
        [args.accelerations]
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

    path = Path(f"evaluation/{model_name}/{model_cfg.id}")
    
    path.mkdir(parents=True, exist_ok=True)

    if args.undersampling:
        path = path / f"{args.accelerations}_{args.mask_type}.json"
    else:
        path = path / "full.json"

    print(f"Evalution file path: {str(path)}")
    # You need to specify your wandb login keys here:
    # wandb.login(key="")
    
    if model_cfg.id is None:
        wandb_logger = WandbLogger(project=model_name, name=run_name, log_model=False, config=OmegaConf.to_container(config, resolve=True))
    else:
        wandb_logger = WandbLogger(project=model_name, name=run_name, log_model=False, config=OmegaConf.to_container(config, resolve=True), id=model_cfg.id, resume="must")

    trainer = pl.Trainer(logger=wandb_logger)
    results = trainer.test(model, dataloaders=dm.val_dataloader())
    wandb.finish()

    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    # if args.undersampling:
    #     path = path + 
    # path = path + f"/{args.accelerations}_{args.mask_type}.json"
    # with open(path, "w") as f:
    #     json.dump(results, f, indent=4)


if __name__ == "__main__":
    args = cli()
    main(args)