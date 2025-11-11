import pytorch_lightning as pl
import numpy as np
import wandb
import matplotlib.pyplot as plt
from modules.transforms import KspaceUNetSample, normalize_to_minus_one_one
from fastmri.evaluate import nmse
from modules.losses import ssim, psnr, LPIPS
from torch.nn.functional import l1_loss
import torch
import random
from fastmri.losses import SSIMLoss


class MRIModule(pl.LightningModule):
    def __init__(self, num_log_images: int = 32):
        super().__init__()
        self.num_log_images = num_log_images
        self.test_log_indices = None
        self.val_log_indices = None
        self.train_log_indices = None

        self.ssim_loss = SSIMLoss()
        self.perc_loss = LPIPS().eval()

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
        # reconstructions = reconstructions.contiguous()
        # targets = targets.contiguous()
        # mse = self.MSE(reconstructions, targets)
        # psnr = self.PSNR(reconstructions, targets)
        # ssim = torch.tensor(1.0) - self.SSIM(reconstructions, targets, max_vals)
        # self.log(f"{flag}/MSE", mse, on_epoch=True, on_step=True, sync_dist=True, batch_size=reconstructions.shape[0])
        # self.log(f"{flag}PSNR", psnr, on_epoch=True, on_step=True, sync_dist=True, batch_size=reconstructions.shape[0])
        # self.log(f"{flag}SSIM", ssim, on_epoch=True, on_step=True, sync_dist=True, batch_size=reconstructions.shape[0])
        
        ssim_loss = self.ssim_loss(reconstructions.unsqueeze(1), targets.unsqueeze(1), max_vals)
        self.log(f"{flag}/SSIM", 1 - ssim_loss, on_epoch=True, on_step=True, sync_dist=True, batch_size=reconstructions.shape[0])
        targets = targets.detach().cpu().numpy()
        reconstructions = reconstructions.detach().cpu().numpy()
        max_vals = max_vals.cpu().numpy()
        SSIM = torch.Tensor(ssim(targets, reconstructions, max_vals)).to(self.device)
        PSNR = torch.Tensor(psnr(targets, reconstructions, max_vals)).to(self.device)
        NMSE = torch.Tensor(nmse(targets, reconstructions)).to(self.device)
        self.log_dict({
            f"{flag}/ssim": SSIM,
            f"{flag}/psnr": PSNR,
            f"{flag}/nmse": NMSE,
        }, on_epoch=True, on_step=True, sync_dist=True, batch_size=reconstructions.shape[0])

    def on_test_batch_end(self, outputs, batch: KspaceUNetSample, batch_idx):
        for k in ("reconstructions", "outputs", "inputs"):
            if k not in outputs.keys():
                raise RuntimeError(f"Expected key {k} in dict returned by test_step")
        
        reconstructions = outputs["reconstructions"]
        targets = batch.target
        max_vals = batch.max_value

        self.log("test/lpips", self.perc_loss(normalize_to_minus_one_one(reconstructions.repeat(1,3,1,1).contiguous()), normalize_to_minus_one_one(targets.repeat(1,3,1,1).contiguous())), on_epoch=True, on_step=True, sync_dist=True, batch_size=reconstructions.shape[0])
        self.log("test/l1_kspace", l1_loss(outputs["outputs"], outputs["inputs"]), on_epoch=True ,on_step=True, sync_dist=True, batch_size=reconstructions.shape[0])
        self.log("test/l1_image", l1_loss(reconstructions, targets), on_epoch=True, on_step=True,  sync_dist=True, batch_size=reconstructions.shape[0])
        self.calculate_metrics(reconstructions, targets, max_vals, "test")
