import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf
from modules.utils import get_from_config
from pathlib import Path
from fastmri.pl_modules import FastMriDataModule
from fastmri.data.subsample import create_mask_for_mask_type
from pathlib import Path
from modules.transforms import KspaceUNetDataTransform

def cli():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    return parser.parse_args()

def main(args):
    torch.set_float32_matmul_precision('high')
    config = OmegaConf.load(args.config)
    model_cfg = config.model
    data_cfg = config.data
    trainer_cfg = config.trainer
    fs_cfg = config.first_stage

    first_stage = get_from_config(fs_cfg)
    try:
        first_stage = first_stage.load_from_checkpoint(fs_cfg.ckpt_path)
    except Exception as e:
        print("Error in loading checkpoint: {e}")
        first_stage = first_stage.load_from_checkpoint(fs_cfg.ckpt_path, strict=False)


    model = get_from_config(model_cfg)
    if model_cfg.ckpt_path is not None:
        print(f"Use Checkpoint: {model_cfg.ckpt_path}")
        model = model.load_from_checkpoint(model_cfg.ckpt_path, first_stage=first_stage)
    else:
        model = model(
            in_channels=model_cfg.params.in_channels, 
            out_channels = model_cfg.params.out_channels, 
            first_stage=first_stage, 
            rescale_latents=model_cfg.params.rescale_latents, 
            normalize_latents=model_cfg.params.normalize_latents
            )
    

    config = {
        "mask_type": "random",
        "center_fractions": [0.04],
        "accelerations": [4],
    }

    mask_func = create_mask_for_mask_type(
        config["mask_type"], config["center_fractions"], config["accelerations"]
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = KspaceUNetDataTransform(mask_func=mask_func, use_seed=False)
    val_transform = KspaceUNetDataTransform(mask_func=mask_func)
    test_transform = KspaceUNetDataTransform()
    data_module = FastMriDataModule(
        data_path=Path("/home/janus/iwbi-cip-datasets/shared/fastMRI/knee/"),
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

if __name__ == "__main__":
   args = cli()
   main(args)