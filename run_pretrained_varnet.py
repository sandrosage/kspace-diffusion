"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
from pathlib import Path
import requests
import torch
from tqdm import tqdm
import json
from fastmri.data import SliceDataset
from fastmri.models import VarNet
import pytorch_lightning as pl
from modules.transforms import  normalize_to_minus_one_one
from fastmri.evaluate import nmse
from modules.losses import ssim, psnr, LPIPS
from torch.nn.functional import l1_loss
from fastmri.data.subsample import create_mask_for_mask_type
import fastmri.data.transforms as T
import matplotlib.pyplot as plt

VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
MODEL_FNAMES = {
    "varnet_knee_mc": "knee_leaderboard_state_dict.pt",
    "varnet_brain_mc": "brain_leaderboard_state_dict.pt",
}

class VARNET(pl.LightningModule):
    def __init__(self, challenge: str = "varnet_knee_mc", state_dict_file: Path = None, output_dir: Path = None):
        super().__init__()

        self.model = VarNet(num_cascades=12, pools=4, chans=18, sens_pools=4, sens_chans=8)
        # download the state_dict if we don't have it
        if state_dict_file is None:
            if not Path(MODEL_FNAMES[challenge]).exists():
                url_root = VARNET_FOLDER
                self._download_model(url_root + MODEL_FNAMES[challenge], MODEL_FNAMES[challenge])

            state_dict_file = MODEL_FNAMES[challenge]

        self.model.load_state_dict(torch.load(state_dict_file))
        self.model = self.model.eval()
        self.perc_loss = LPIPS().eval()

        self.ssim_lst = []
        self.nmse_list = []
        self.lpips_list = []
        self.pnsr_list = []

        self.output_dir = output_dir

    @staticmethod
    def _download_model(url, fname):
        response = requests.get(url, timeout=10, stream=True)

        chunk_size = 8 * 1024 * 1024  # 8 MB chunks
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        progress_bar = tqdm(
            desc="Downloading state_dict",
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
        )

        with open(fname, "wb") as fh:
            for chunk in response.iter_content(chunk_size):
                progress_bar.update(len(chunk))
                fh.write(chunk)
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            output, slice_num, fname = self.run_varnet_model(batch)
        
        target = batch.target
        max_value = batch.max_value
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
        
        self.log_dict(metrics, on_epoch=True, on_step=True, sync_dist=True, batch_size=output.shape[0])

        if batch_idx % 40:

            if self.output_dir is not None:

                plt.imshow(output.squeeze(0), cmap="gray")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(self.output_dir / ("new_" + str(fname[:-3]) +  "_" + str(slice_num) + ".png") , bbox_inches="tight", dpi=1000, pad_inches=0)
                plt.close()

    def on_test_epoch_end(self):
        for metric,label in zip([self.ssim_lst, self.nmse_list, self.pnsr_list, self.lpips_list], ["ssim", "nmse", "psnr", "lpips"]):
            epoch_metric = torch.stack(metric)
            mean = epoch_metric.mean()
            std = epoch_metric.std(unbiased=True)
            self.log(label + "_mean", mean, on_epoch=True)
            self.log(label + "_std", std, on_epoch=True)
            metric.clear()

    def run_varnet_model(self, batch):
        crop_size = batch.crop_size
        masked_kspace = batch.masked_kspace.unsqueeze(1)
        mask = batch.mask.unsqueeze(1)
        output = self.model(masked_kspace.to(self.device), mask.to(self.device))

        # detect FLAIR 203
        if output.shape[-1] < crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])

        # output = T.center_crop(output, crop_size)[0]
        output = T.center_crop(output, crop_size)
        return output, int(batch.slice_num[0]), batch.fname[0]
    

def create_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--challenge",
        default="varnet_knee_mc",
        choices=(
            "varnet_knee_mc",
            "varnet_brain_mc",
        ),
        type=str,
        help="Model to run",
    )

    parser.add_argument(
        "--state_dict_file",
        default=None,
        type=Path,
        help="Path to saved state_dict (will download if not provided)",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to subsampled data",
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

    path = f"evaluation/VarNet/{args.mask_type}_{args.accelerations}_varnet.json"

    print(path)
    output_dir = None
    # args.output_path = /home/atuin/b180dc/b180dc46/VarNet
    if args.output_path is not None:
        output_dir = args.output_path  /str(args.mask_type) / str(args.accelerations) 
        output_dir.mkdir(parents=True, exist_ok=True)

    mask_func = create_mask_for_mask_type(
        args.mask_type, [args.center_fractions], [args.accelerations]
    )

    data_transform = T.VarNetDataTransform(mask_func=mask_func)

    dataset = SliceDataset(
        root=args.data_path, transform=data_transform, challenge="singlecoil"
    )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)

    model = VARNET(args.challenge, args.state_dict_file, output_dir)

    trainer = pl.Trainer()

    results = trainer.test(model, dataloaders=dataloader)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)