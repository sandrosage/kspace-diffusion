import pytorch_lightning as pl
import numpy as np
import wandb
import matplotlib.pyplot as plt
from fastmri.losses import SSIMLoss
from torchmetrics.image import PeakSignalNoiseRatio
from matplotlib import colors
import fastmri.data.transforms as T
from modules.transforms import KspaceUNetSample
import torch

np.random.seed(111)

class MRIModule(pl.LightningModule):
    def __init__(self, num_log_images: int = 16):
        super().__init__()
        self.num_log_images = num_log_images
        self.test_log_indices = None
        self.val_log_indices = None
        self.train_log_indices = None

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
            self.log_image(outputs["fname"], batch_idx, outputs["slice_num"], outputs["target"].squeeze(0), outputs["rec_img"].squeeze(0), "val")
    
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
            self.log_image(outputs["fname"], batch_idx, outputs["slice_num"], outputs["target"].squeeze(0), outputs["rec_img"].squeeze(0), "test")
    
    def on_train_batch_end(self, outputs, batch, batch_idx):

        if self.train_log_indices is None:
            self.train_log_indices = list(
                np.random.permutation(len(self.trainer.train_dataloader))[
                    : 128
                ]
            )
        self.train_log_indices.append(1)
        
        if batch_idx in self.train_log_indices:
            input = np.squeeze(T.tensor_to_complex_np(outputs["input"].cpu().detach()), axis=0)
            reconstruction = np.squeeze(T.tensor_to_complex_np(outputs["reconstruction"].cpu().detach()), axis=0)
            # np.save(str(batch_idx) + ".npy", reconstruction)
            rec_img = outputs["rec_img"].squeeze(0).detach()
            target = outputs["target"].squeeze(0).detach()

            self.store_kspace(batch.fname, batch_idx, batch.slice_num, input, reconstruction, "train")
            self.log_image(batch.fname, batch_idx, batch.slice_num, target, rec_img, "train")

    def store_kspace(self, fname, batch_idx, slice_num, input, reconstruction, flag):
            # Create figure and axes
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Display images
            axes[0].imshow(np.log(np.abs(input) + 1e-9))
            axes[0].set_title("Kspace")

            axes[1].imshow(np.log(np.abs(reconstruction) + 1e-9))
            axes[1].set_title("Reconstruction")

            # Show the figure
            plt.tight_layout()
            # fig.savefig('images/{}_{}_{}_{}_Grid.png'.format(flag, fname[0][:-3], batch_idx, str(slice_num.cpu().numpy()[0])), dpi=300)
            self.logger.experiment.log({'kspaces/{}_{}_{}_{}_Grid.png'.format(flag, fname[0][:-3], batch_idx, str(slice_num.cpu().numpy()[0])) : wandb.Image(plt)})
            plt.clf()
            plt.cla()
            plt.close()

    def log_image(self, fname, batch_idx, slice_num, target, rec_img, flag):
        target = target.cpu().numpy()
        rec_img = rec_img.cpu().numpy()
        diff = (target - rec_img)
        fig, ax = plt.subplots(1,3,figsize=(18,5))
        fig.subplots_adjust(wspace=0.0)
        # orig
        ax[0].imshow(target,'gray')
        ax[0].set_title("Target")
        # reconstructed
        ax[1].imshow(rec_img,'gray')
        ax[1].set_title("Reconstructed")
        # difference
        ax[2].imshow(diff,'inferno',norm=colors.Normalize(vmin=diff.min(), vmax=diff.max()))
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
        self.logger.experiment.log({'images/{}_{}_{}_{}_Grid.png'.format(flag, fname[0][:-3], batch_idx, str(slice_num.cpu().numpy()[0])) : wandb.Image(plt)})
        plt.clf()
        plt.cla()
        plt.close()

class NewMRIModule(pl.LightningModule):
    def __init__(self, num_log_images: int = 32):
        super().__init__()
        self.num_log_images = num_log_images
        self.test_log_indices = None
        self.val_log_indices = None
        self.train_log_indices = None

        self.SSIM = SSIMLoss()
        self.PSNR = PeakSignalNoiseRatio()
    
    def on_train_batch_end(self, outputs, batch: KspaceUNetSample, batch_idx):

        self.calculate_metrics(batch, outputs, "train")

        if self.train_log_indices is None:
            self.train_log_indices = list(
                np.random.permutation(len(self.trainer.train_dataloader))[
                    : self.num_log_images - 1
                ]
            )
        self.train_log_indices.insert(0, 1)
        
        if batch_idx in self.train_log_indices:
            undersampled_mri_image = outputs["undersampled_mri_image"].squeeze(0).detach().cpu().numpy()
            reconstructed_mri_image = outputs["reconstructed_mri_image"].squeeze(0).detach().cpu().numpy()
            full_mri_image = outputs["full_mri_image"].squeeze(0).detach().cpu().numpy()
            self.log_image(batch.fname, batch_idx, batch.slice_num, undersampled_mri_image, reconstructed_mri_image, full_mri_image, "train")
    
    def on_validation_batch_end(self, outputs, batch: KspaceUNetSample, batch_idx):

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
            self.val_log_indices = list(1) + list(
                np.random.permutation(len(self.trainer.val_dataloaders))[
                    : self.num_log_images - 1
                ]
            )
        
        if batch_idx in self.val_log_indices:
            input = outputs["input"]
            output = outputs["output"]
            target = batch.target
            print(input.shape, output.shape, target.shape)

            error = torch.abs(target - output)
            output = output / output.max()
            target = target / target.max()
            error = error / error.max()


            undersampled_mri_image = outputs["input"].squeeze(0)
            reconstructed_mri_image = outputs["output"].squeeze(0).detach().cpu().numpy()
            full_mri_image = outputs["target"].squeeze(0).detach().cpu().numpy()
            self.log_image(batch.fname, batch_idx, batch.slice_num, undersampled_mri_image, reconstructed_mri_image, full_mri_image, "val")

    
    def log_image(self, fname, batch_idx, slice_num, undersampled_mri_image, reconstructed_mri_image, full_mri_image, flag):
        fig, ax = plt.subplots(1,3,figsize=(18,5))
        fig.subplots_adjust(wspace=0.0)
    
        ax[0].imshow(undersampled_mri_image,'gray')
        ax[0].set_title("Undersampled Mri Image")

        ax[1].imshow(reconstructed_mri_image,'gray')
        ax[1].set_title("Reconstructed Mri Image")

        ax[2].imshow(full_mri_image, 'gray')
        ax[2].set_title("Full Mri Image")
        
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
        self.logger.experiment.log({'images/{}/{}_{}_{}_{}_Grid.png'.format(self.trainer.current_epoch, flag, fname[0][:-3], batch_idx, str(slice_num.cpu().numpy()[0])) : wandb.Image(plt)})
        plt.clf()
        plt.cla()
        plt.close()
    
    def calculate_metrics(self, batch: KspaceUNetSample, outputs, flag:str):
        undersampled_mri_image = outputs["undersampled_mri_image"]
        reconstructed_mri_image = outputs["reconstructed_mri_image"]
        full_mri_image = outputs["full_mri_image"]
        max_val = batch.max_value
        self.log_dict({
            f"{flag}/ssim_loss": self.SSIM(reconstructed_mri_image, full_mri_image, data_range=max_val),
        }, on_epoch=True, on_step=True, sync_dist=True, batch_size=undersampled_mri_image.shape[0])
