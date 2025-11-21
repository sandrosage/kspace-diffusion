import argparse

from pathlib import Path
import json
import torch
from tqdm import tqdm
import fastmri.data.transforms as T
from fastmri.data import SliceDataset
from fastmri.models import Unet
import pytorch_lightning as pl
from modules.transforms import  normalize_to_minus_one_one, KspaceUNetDataTransform, kspace_to_mri
from fastmri.evaluate import nmse
from modules.losses import ssim, psnr, LPIPS
from torch.nn.functional import l1_loss
from fastmri.data.subsample import create_mask_for_mask_type
from pl_modules.ldm_module import LDM
from pl_modules.diffusers_vae_module import WeightedSSIMKspaceAutoencoderKL
from modules.cgs import ConsistencyGuidanceSampler
import matplotlib.pyplot as plt

class LDM_EVAL(pl.LightningModule):
    def __init__(self, model_path: Path, first_stage_path: Path, timesteps: int, output_dir: Path = None, cgs: bool = False):
        super().__init__()

        self.first_stage = WeightedSSIMKspaceAutoencoderKL.load_from_checkpoint(first_stage_path).eval()
        self.model = LDM.load_from_checkpoint(model_path, first_stage=self.first_stage).eval()
        self.cgs = ConsistencyGuidanceSampler(ldm=self.model, scheduler=self.model.scheduler,num_inference_steps=timesteps, cgs=cgs)
        self.perc_loss = LPIPS().eval()

        self.ssim_lst = []
        self.nmse_list = []
        self.lpips_list = []
        self.pnsr_list = []

        self.output_dir = output_dir
        print(f"Inference timesteps: {timesteps}")


    def test_step(self, batch, batch_idx):
        output_kspace = self.cgs(batch)
        output = kspace_to_mri(output_kspace)
        target = batch.target
        max_value = batch.max_value

        metrics = {
            "l1_loss": l1_loss(output_kspace, batch.full_kspace),
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

        if batch_idx % 40:

            if self.output_dir is not None:
                plt.imshow(output.squeeze(0), cmap="gray")    
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(self.output_dir / (str(batch.fname[0][:-3]) +  "_" + str(batch.slice_num.item()) + ".png") , bbox_inches="tight", dpi=1000, pad_inches=0)
                plt.close()

        self.log_dict(metrics, on_epoch=True, on_step=True, sync_dist=True, batch_size=output.shape[0])
    
    def on_test_epoch_end(self):
        for metric,label in zip([self.ssim_lst, self.nmse_list, self.pnsr_list, self.lpips_list], ["ssim", "nmse", "psnr", "lpips"]):
            epoch_metric = torch.stack(metric)
            mean = epoch_metric.mean()
            std = epoch_metric.std(unbiased=True)
            self.log(label + "_mean", mean, on_epoch=True)
            self.log(label + "_std", std, on_epoch=True)
            metric.clear()

def create_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--cgs",
        action="store_true",
        help="flag for using CGS in diffusion sampling"
    )

    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Model to run",
    )

    parser.add_argument(
        "--model_path",
        default="LDM_/xfv8d48c/checkpoints/last.ckpt",
        type=Path,
        help="Path to ldm model checkpoints file",
    )

    parser.add_argument(
        "--first_stage_path",
        default="WeightedSSIMKspaceAutoencoderKL/b63zsecl/checkpoints/WeightedSSIMKspaceAutoencoderKL-epoch=66.ckpt",
        type=Path,
        help="Path to first stage checkpoints file",
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
        "--store_files",
        action="store_true",
        help="Path flag to store the output files"
    )

    parser.add_argument(
        "--timesteps",
        default=50,
        type=int,
        help="number of inference timesteps"
    )

    return parser

if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    
    if args.cgs:
        path = f"evaluation/Ours/new_cgs_{args.mask_type}_{args.accelerations}_{args.timesteps}_ours.json"
    else:
        path = f"evaluation/Ours/new_{args.mask_type}_{args.accelerations}_{args.timesteps}_ours.json"

    if args.store_files:
        output_path = Path("/home/atuin/b180dc/b180dc46/LDM")
        print(f"Store files: {args.store_files}")
    else:
        output_path = None

    print(path)

    print(f"Use CGS: {args.cgs}")

    output_dir = None
    if output_path is not None:
        if args.cgs:
            output_dir = output_path / str(args.mask_type) / str(args.accelerations) / str(args.timesteps) / "cgs"
        else:
            output_dir = output_path / str(args.mask_type) / str(args.accelerations) / str(args.timesteps) / "no_cgs"
        output_dir.mkdir(parents=True, exist_ok=True)

    model = LDM_EVAL(args.model_path, args.first_stage_path, args.timesteps, output_dir, args.cgs)



    mask_func = create_mask_for_mask_type(
        args.mask_type, [args.center_fractions], [args.accelerations]
    )
    data_transform = KspaceUNetDataTransform(mask_func=mask_func)

    dataset = SliceDataset(
            root=args.data_path,
            transform=data_transform,
            challenge="singlecoil",
        )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)
    trainer = pl.Trainer(inference_mode=False)
    results = trainer.test(model, dataloaders=dataloader)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)