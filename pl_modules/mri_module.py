import pytorch_lightning as pl
import numpy as np
import wandb
import matplotlib.pyplot as plt
from fastmri.losses import SSIMLoss
from torchmetrics.image import PeakSignalNoiseRatio
from matplotlib import colors

class MRIModule(pl.LightningModule):
    def __init__(self, num_log_images: int = 16):
        super().__init__()
        self.num_log_images = num_log_images
        self.test_log_indices = None
        self.val_log_indices = None

        self.SSIM = SSIMLoss()
        self.PSNR = PeakSignalNoiseRatio()

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        ssim_loss = self.SSIM(outputs["target"], outputs["rec_img"], outputs["max_value"])
        psnr_loss = self.PSNR(outputs["target"], outputs["rec_img"])

        metrics = {
            "val/ssim_loss": ssim_loss,
            "val/psnr_loss": psnr_loss
        }
        self.log_dict(metrics, sync_dist=True, on_epoch=True, on_step=True, batch_size=batch.masked_kspace.shape[0])

        if self.val_log_indices is None:
            self.val_log_indices = list(
                np.random.permutation(len(self.trainer.val_dataloaders))[
                    : self.num_log_images
                ]
            )
        if batch_idx in self.val_log_indices:
            self.log_image(outputs["fname"], batch_idx, outputs["slice_num"], outputs["target"], outputs["rec_img"], "val")
    
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        ssim_loss = self.SSIM(outputs["target"], outputs["rec_img"], outputs["max_value"])
        psnr_loss = self.PSNR(outputs["target"], outputs["rec_img"])

        metrics = {
            "test/ssim_loss": ssim_loss,
            "test/psnr_loss": psnr_loss
        }
        self.log_dict(metrics, sync_dist=True, on_epoch=True, on_step=True, batch_size=batch.masked_kspace.shape[0])

        if self.test_log_indices is None:
            self.test_log_indices = list(
                np.random.permutation(len(self.trainer.test_dataloaders))[
                    : self.num_log_images
                ]
            )
        if batch_idx in self.test_log_indices:
            self.log_image(outputs["fname"], batch_idx, outputs["slice_num"], outputs["target"], outputs["rec_img"], "test")

    def log_image(self, fname, batch_idx, slice_num, target, rec_img, flag):
        target = target.cpu().numpy()
        rec_img = rec_img.cpu().numpy()
        diff = np.abs(target - rec_img)
        fig, ax = plt.subplots(1,3,figsize=(18,5))
        fig.subplots_adjust(wspace=0.0)
        # orig
        ax[0].imshow(target,'gray')
        ax[0].set_title("Target")
        # reconstructed
        ax[1].imshow(rec_img,'gray')
        ax[1].set_title("Reconstructed")
        # difference
        ax[2].imshow(diff,'inferno',norm=colors.Normalize(vmin=0, vmax=diff.max()+.01))
        ax[2].set_title("Difference")
        
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
        self.logger.experiment.log({'{}/images/{}_{}_{}_Grid.png'.format(flag, fname, batch_idx, slice_num) : wandb.Image(plt)})
        plt.clf()
        plt.cla()
        plt.close()