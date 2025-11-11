from pathlib import Path
from modules.utils import get_from_config
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.mri_data import SliceDataset
from argparse import ArgumentParser
from modules.transforms import KspaceUNetDataTransform
import pytorch_lightning as pl
import torch

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

def load_model(ckpt_path: Path, undersampling: bool = False):
    model = get_from_config(ckpt_path)
    model = model.load_from_checkpoint(ckpt_path, strict=False, undersampling=undersampling)
    for p in model.parameters():
        p.requires_grad = False
    return model

def cli():
    parser = ArgumentParser()
    parser.add_argument("--undersampling", action="store_true", help="Flag for undersampled k-space")
    parser.add_argument("--accelerations", type=int, default=4, help="Acceleration factor for undersampling mask")
    return parser.parse_args()

if __name__ == "__main__":

    args = cli()

    mask_func = create_mask_for_mask_type(
        "equispaced", [0.08], [args.accelerations]
    )
    data_transform = KspaceUNetDataTransform(mask_func=mask_func)

    dataset = SliceDataset(
            root=Path("/home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val"),
            transform=data_transform,
            challenge="singlecoil",
        )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)
    model = load_model(ckpt_path=args.ckpt_path, undersampling=args.undersampling)
    for batch in dataloader:

