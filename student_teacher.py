import torch 
import torch.nn.functional as F
from modules.unet import Unet_FastMRI
from pl_modules.myUnet_module import MyUnetModule, NewMRIModule
from fastmri.pl_modules import FastMriDataModule
from modules.transforms import KspaceUNetSample, kspace_to_mri, KspaceUNetDataTransform
from typing import Tuple
import wandb
from torch.nn import L1Loss
from datetime import datetime 
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from fastmri.data.subsample import create_mask_for_mask_type
from pathlib import Path
import matplotlib.pyplot as plt

class StudentNoSkipUnet(NewMRIModule):
    def __init__(self, path: str, num_log_images = 128):
        super().__init__(num_log_images)
        self.teacher = MyUnetModule.load_from_checkpoint(path)
        self.teacher.eval()
        self.teacher.model.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.lr = 0.0003
        self.lr_step_size = 40
        self.lr_gamma = 0.1
        self.weight_decay = 0.0
        self.model = Unet_FastMRI(2, 2, 128, with_residuals=False)
    
    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.model(input)
        return output
    
    def shared_step(self, batch: KspaceUNetSample) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        full_kspace = batch.full_kspace
        masked_kspace = batch.masked_kspace
        mask = batch.mask
        full_mri_image = kspace_to_mri(full_kspace, (320,320))
        undersampled_mri_image = kspace_to_mri(masked_kspace, (320,320))
        return full_kspace, masked_kspace, full_mri_image, undersampled_mri_image
    
    def training_step(self, batch: KspaceUNetSample, batch_idx):
        full_kspace, masked_kspace, full_mri_image, _ = self.shared_step(batch)
        input = masked_kspace.permute(0,3,1,2).contiguous()
        input, mean, std = self.norm(input)
        # teacher output generation
        z, stack = self.teacher.downsample(input)
        teacher_output = self.teacher.upsample(z, stack)

        # student output generation
        student_output = self(input)

        teacher_output = self.unnorm(teacher_output, mean, std)
        student_output = self.unnorm(student_output, mean, std)

        teacher_output = teacher_output.permute(0,2,3,1).contiguous()
        student_output = student_output.permute(0,2,3,1).contiguous()

        teacher_mri_output = kspace_to_mri(teacher_output, (320,320))
        student_mri_output = kspace_to_mri(student_output, (320,320))
        loss = self.reconstruction_distillation_loss(student_mri_output, teacher_mri_output, full_mri_image)
        self.log("train/dist_loss", loss, on_epoch=True, on_step=True, prog_bar=True, batch_size=full_kspace.shape[0], sync_dist=True)
        return {
            "loss": loss, 
            "undersampled_mri_image": teacher_mri_output,
            "reconstructed_mri_image": student_mri_output,
            "full_mri_image": full_mri_image
        }
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    def reconstruction_distillation_loss(self, student_output, teacher_output, target, alpha=0.5):
        # recon_loss = F.mse_loss(student_output, target)
        distill_loss = F.l1_loss(student_output, teacher_output)
        # return alpha * distill_loss + (1 - alpha) * recon_loss
        return distill_loss


if __name__ == "__main__":
    print("Starting...")
    config = {
        "mask_type": "equispaced_fraction",
        "center_fractions": [0.04],
        "accelerations": [8],
        "loss_domain": "ssim", 
        "criterion": L1Loss(),
        "n_channels": 128,
        "with_residual": False,
        "latent_dim": 128,
        "mode": "interpolation"
    }

    model_name = "StudentNoSkipUnet"
    run_name = model_name + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    wandb.login(key="c210746318a0cf3a3fb1d542db1864e0a789e94c")
    wandb_logger = WandbLogger(project="Kspace-Experiments", name=run_name, log_model=True, config=config)

    model = StudentNoSkipUnet(path="6jxi7pym/myunet-epoch=01.ckpt")

    model_checkpoint = ModelCheckpoint(
        save_top_k=2,
        monitor="val/ssim_loss_epoch",
        mode="min",
        filename="myunet-{epoch:02d}"
    )
    trainer = pl.Trainer(devices=1, max_epochs=2, logger=wandb_logger, callbacks=[model_checkpoint])

    mask_func = create_mask_for_mask_type(
        config["mask_type"], config["center_fractions"], config["accelerations"]
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = KspaceUNetDataTransform(mask_func=mask_func, use_seed=False)
    val_transform = KspaceUNetDataTransform(mask_func=mask_func)
    test_transform = KspaceUNetDataTransform()
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=Path("/vol/datasets/cil/2021_11_23_fastMRI_data/knee/unzipped"),
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
    )
    trainer.fit(model,data_module.train_dataloader())
    wandb.finish()
    model = MyUnetModule.load_from_checkpoint("6jxi7pym/myunet-epoch=01.ckpt")
    # train_dataloader = data_module.train_dataloader()
    # for i in range(5):
    #     batch = next(iter(train_dataloader))
    #     full_kspace = batch.full_kspace
    #     masked_kspace = batch.masked_kspace
    #     masked_kspace = masked_kspace.to("cuda")
    #     mask = batch.mask
    #     mask = mask.to("cuda")
    #     input = masked_kspace.permute(0,3,1,2).contiguous()
    #     input = input.to("cuda")
    #     input, mean, std = model.norm(input)
    #     z, stack = model.downsample(input)
    #     output = model.upsample(z, stack)
    #     output = model.unnorm(output, mean, std)
    #     reconstructed_kspace = output.permute(0,2,3,1).contiguous()
    #     zero = torch.zeros(1, 1, 1, 1).to("cuda")
    #     soft_dc = torch.where(mask, reconstructed_kspace - masked_kspace, zero)
    #     output = reconstructed_kspace - soft_dc
    #     full_mri_image = kspace_to_mri(full_kspace, (320,320))
    #     afterdc_mri_image = kspace_to_mri(output, (320,320))
    #     reconstructed_mri_image = kspace_to_mri(reconstructed_kspace, (320,320))
    #     masked_mri_image = kspace_to_mri(masked_kspace, (320,320))
    #     masked_mri_image = masked_mri_image.squeeze(0).detach().cpu().numpy()
    #     reconstructed_mri_image = reconstructed_mri_image.squeeze(0).detach().cpu().numpy()
    #     afterdc_mri_image = afterdc_mri_image.squeeze(0).detach().cpu().numpy()
    #     full_mri_image = full_mri_image.squeeze(0).detach().cpu().numpy()
    #     fig, ax = plt.subplots(1,4,figsize=(18,5))
    #     fig.subplots_adjust(wspace=0.0)
    
    #     ax[0].imshow(full_mri_image,'gray')
    #     ax[0].set_title("Full Mri Image")

    #     ax[1].imshow(reconstructed_mri_image,'gray')
    #     ax[1].set_title("Reconstructed Mri Image")

    #     ax[2].imshow(afterdc_mri_image, 'gray')
    #     ax[2].set_title("After dc Mri Image")

    #     ax[3].imshow(masked_mri_image, 'gray')
    #     ax[3].set_title("Undersampled Mri Image")

        
    #     # remove all the ticks (both axes), and tick labels
    #     for axes in ax:
    #         axes.set_xticks([])
    #         axes.set_yticks([])
    #     # remove the frame of the chart
    #     for axes in ax:
    #         axes.spines['top'].set_visible(False)
    #         axes.spines['right'].set_visible(False)
    #         axes.spines['bottom'].set_visible(False)
    #         axes.spines['left'].set_visible(False)
    #     # remove the white space around the chart
    #     plt.tight_layout()
    #     plt.savefig(f"{i}.png")
    # print("Done")