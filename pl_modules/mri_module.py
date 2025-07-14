import pytorch_lightning as pl
import numpy as np
import wandb
import matplotlib.pyplot as plt
from modules.transforms import KspaceUNetSample
from fastmri.evaluate import nmse
from modules.losses import ssim, psnr
import torch
import random

class MRIModule(pl.LightningModule):
    def __init__(self, num_log_images: int = 32):
        super().__init__()
        self.num_log_images = num_log_images
        self.test_log_indices = None
        self.val_log_indices = None
        self.train_log_indices = None

    
    def on_train_batch_end(self, outputs, batch: KspaceUNetSample, batch_idx):

        for k in ("reconstructions",):
            if k not in outputs.keys():
                raise RuntimeError(f"Expected key {k} in dict returned by training_step")

        reconstructions = outputs["reconstructions"].detach()
        targets = batch.target
        max_vals = batch.max_value

        self.calculate_metrics(reconstructions, targets, max_vals, "train")

        if self.train_log_indices is None:
            self.train_log_indices = list([1]) + list(
                np.random.permutation(len(self.trainer.train_dataloader))[
                    : self.num_log_images - 1
                ]
            )
        
        if batch_idx in self.train_log_indices:
            idx = random.sample(range(reconstructions.size(0)), 1)[0]
            target = targets[idx]
            reconstruction = reconstructions[idx]
            slice_num = batch.slice_num[idx]
            fname = batch.fname[idx]
            error = torch.abs(target - reconstruction)
            reconstruction = reconstruction / reconstruction.max()
            target = target / target.max()
            error = error / error.max()
            self.log_image(fname, batch_idx, slice_num, target, reconstruction, error, "train")
        
    
    def on_validation_batch_end(self, outputs, batch: KspaceUNetSample, batch_idx):
        for k in ("reconstructions",):
            if k not in outputs.keys():
                raise RuntimeError(f"Expected key {k} in dict returned by validation_step")
        
        reconstructions = outputs["reconstructions"]
        targets = batch.target
        max_vals = batch.max_value

        self.calculate_metrics(reconstructions, targets, max_vals, "val")

        for k in (
            "input",
            "output",
            "loss"
        ):
            if k not in outputs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by validation_step."
                )
            
        if outputs["input"].ndim == 2:
            outputs["input"] = outputs["input"].unsqueeze(0)
        elif outputs["input"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        
        if outputs["output"].ndim == 2:
            outputs["output"] = outputs["output"].unsqueeze(0)
        elif outputs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
    
        if self.val_log_indices is None:
            self.val_log_indices = list([1]) + list(
                np.random.permutation(len(self.trainer.val_dataloaders))[
                    : self.num_log_images - 1
                ]
            )
        
        if batch_idx in self.val_log_indices:
            idx = random.sample(range(reconstructions.size(0)), 1)[0]
            target = targets[idx]
            reconstruction = reconstructions[idx]
            slice_num = batch.slice_num[idx]
            fname = batch.fname[idx]
            error = torch.abs(target - reconstruction)
            reconstruction = reconstruction / reconstruction.max()
            target = target / target.max()
            error = error / error.max()
            self.log_image(fname, batch_idx, slice_num, target, reconstruction, error, "val")

    
    def log_image(self, fname, batch_idx, slice_num, target, reconstruction, error, flag):
        target = target.squeeze(0).detach().cpu().numpy()
        reconstruction = reconstruction.squeeze(0).detach().cpu().numpy()
        error = error.squeeze(0).detach().cpu().numpy()
        fig, ax = plt.subplots(1,3,figsize=(18,5))
        fig.subplots_adjust(wspace=0.0)
    
        ax[0].imshow(target,'gray')
        ax[0].set_title("Target")

        ax[1].imshow(reconstruction,'gray')
        ax[1].set_title("Reconstruction")

        ax[2].imshow(error, 'inferno')
        ax[2].set_title("Error")
        
        # remove all the ticks (both axes), and tick labels
        for axes in ax:
            axes.set_xticks([])
            axes.set_yticks([])
        # remove the frame of the chart
        for axes in ax:
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)
            axes.spines['bottom'].set_visible(False)
            axes.spines['left'].set_visible(False)
        # remove the white space around the chart
        plt.tight_layout()
        self.logger.experiment.log({'images/{}/{}_{}_{}_{}_Grid.png'.format(self.trainer.current_epoch, flag, fname[:-3], batch_idx, str(slice_num.cpu().numpy())) : wandb.Image(plt)})
        plt.clf()
        plt.cla()
        plt.close()
    
    def calculate_metrics(self, reconstructions: torch.Tensor, targets: torch.Tensor, max_vals: torch.Tensor, flag:str):
        targets = targets.cpu().numpy()
        reconstructions = reconstructions.cpu().numpy()
        max_vals = max_vals.cpu().numpy()
        SSIM = torch.Tensor(ssim(targets, reconstructions, max_vals))
        PSNR = torch.Tensor(psnr(targets, reconstructions, max_vals))
        NMSE = torch.Tensor(nmse(targets, reconstructions))
        self.log_dict({
            f"{flag}/ssim": SSIM,
            f"{flag}/psnr": PSNR,
            f"{flag}/nmse": NMSE,
        }, on_epoch=True, on_step=True, sync_dist=True, batch_size=reconstructions.shape[0])
