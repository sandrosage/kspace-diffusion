"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import json
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import fastmri
import fastmri.data.transforms as T
import pytorch_lightning as pl
from typing import Optional, Dict, Tuple
import torch
from fastmri.data.subsample import MaskFunc, create_mask_for_mask_type
from fastmri.data import SliceDataset
from modules.transforms import  normalize_to_minus_one_one
from fastmri.evaluate import nmse
from modules.losses import ssim, psnr, LPIPS
from torch.nn.functional import l1_loss
import matplotlib.pyplot as plt

class ZeroFilledDataTransform:
    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc],
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed
    
    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, str, int, float]:
        
        if target is not None:
            target_torch = T.to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = T.to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])


        masked_kspace, mask_torch, num_low_frequencies = T.apply_mask(
            kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
        )

        image = fastmri.ifft2c(masked_kspace)
        
        image = T.complex_center_crop(image, (320,320))

        image = fastmri.complex_abs(image)
    
        return (image, target_torch, fname, slice_num, max_value)

class ZEROFILLED(pl.LightningModule):
    def __init__(self, output_dir: Path = None):
        super().__init__()

        self.perc_loss = LPIPS().eval()
        self.ssim_lst = []
        self.nmse_list = []
        self.lpips_list = []
        self.pnsr_list = []

        self.output_dir = output_dir
    
    def test_step(self, batch, batch_idx):
        output, target, fname, slice_num, max_value = batch
        
        if not (slice_num.item() < 5):

            metrics = {
                "l1_loss": l1_loss(output, target),
                "lpips": self.perc_loss(normalize_to_minus_one_one(output.unsqueeze(0).repeat(1,3,1,1).contiguous()), normalize_to_minus_one_one(target.unsqueeze(0).repeat(1,3,1,1).contiguous()))
            }
            target = target.cpu().numpy()
            output = output.cpu().numpy()
            max_value = max_value.cpu().numpy()
            metrics["ssim"] = torch.tensor(ssim(target, output, max_value)).to(self.device)
            metrics["nmse"] = torch.tensor(nmse(target, output)).to(self.device)
            metrics["psnr"] = torch.tensor(psnr(target, output, max_value)).to(self.device)

            self.ssim_lst.append(metrics["ssim"].detach().cpu())
            self.nmse_list.append(metrics["nmse"].detach().cpu())
            self.pnsr_list.append(metrics["psnr"].detach().cpu())
            self.lpips_list.append(metrics["lpips"].detach().cpu())
            self.log_dict(metrics, on_epoch=True, on_step=False, sync_dist=True, batch_size=output.shape[0])

            if batch_idx % 40:

                if self.output_dir is not None:

                    plt.imshow(output.squeeze(0), cmap="gray")
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(self.output_dir / (str(fname[0][:-3]) +  "_" + str(slice_num.item()) + ".png") , bbox_inches="tight", dpi=1000, pad_inches=0)
                    plt.close()

                    plt.imshow(target.squeeze(0), cmap="gray")
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(self.output_dir / ("gt_" + str(fname[0][:-3]) +  "_" + str(slice_num.item()) + ".png") , bbox_inches="tight", dpi=1000, pad_inches=0)
                    plt.close()
    
    def on_test_epoch_end(self):
        for metric,label in zip([self.ssim_lst, self.nmse_list, self.pnsr_list, self.lpips_list], ["ssim", "nmse", "psnr", "lpips"]):
            epoch_metric = torch.stack(metric)
            mean = epoch_metric.mean()
            std = epoch_metric.std(unbiased=True)
            self.log(label + "_mean", mean, on_epoch=True)
            self.log(label + "_std", std, on_epoch=True)
            metric.clear()
        
        
        


def create_arg_parser():
    parser = ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to the data",
    )

    parser.add_argument(
        "--accelerations",
        type=int,
        required=False,
        help="Acceleration factor 4/6/8",
        default=4
    )

    parser.add_argument(
        "--mask_type",
        type=str,
        required=False,
        help="Mask function: random, equispaced, etc.",
        default="equispaced"
    )

    parser.add_argument(
        "--center_fractions",
        type=int,
        required=False,
        help="Center fractions: 0.04/0.08",
        default=0.08
    )

    parser.add_argument(
        "--output_path",
        default=None,
        type=Path,
        required=False,
        help="Path where to store the output files"
    )

    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    
    path = f"evaluation/ZF/{args.mask_type}_{args.accelerations}_zero_filled.json"

    print(path)
    args.output_path = None
    # args.output_path = "/home/atuin/b180dc/b180dc46/ZF"
    output_dir = None
    if args.output_path is not None:
        output_dir = args.output_path  /str(args.mask_type) / str(args.accelerations) 
        output_dir.mkdir(parents=True, exist_ok=True)

    mask_func = create_mask_for_mask_type(
        args.mask_type, [args.center_fractions], [args.accelerations]
    )

    data_transform = ZeroFilledDataTransform(which_challenge="singlecoil", mask_func=mask_func)

    dataset = SliceDataset(
            root=args.data_path,
            transform=data_transform,
            challenge="singlecoil",
        )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)

    model = ZEROFILLED(output_dir=output_dir)
    trainer = pl.Trainer()
    results = trainer.test(model, dataloaders=dataloader)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)