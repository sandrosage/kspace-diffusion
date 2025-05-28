import torch 
import torch.nn.functional as F
from modules.unet import Unet_FastMRI
from pl_modules.myUnet_module import MyUnetModule, NewMRIModule
from modules.transforms import KspaceUNetSample, kspace_to_mri
from typing import Tuple

class StudentNoSkipUnet(NewMRIModule):
    def __init__(self, path: str, num_log_images = 32):
        super().__init__(num_log_images)
        self.teacher = MyUnetModule.load_from_checkpoint(path)
        self.teacher.eval()
        self.model = Unet_FastMRI(2, 2, 128, with_residuals=False)
    
    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        input = masked_kspace.permute(0,3,1,2).contiguous()
        input, mean, std = self.norm(input)
        output = self.model(input)
        output = self.unnorm(output, mean, std)
        output = output.permute(0,2,3,1).contiguous()
        return output
    
    def shared_step(self, batch: KspaceUNetSample) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # full_kspace = complex_center_crop(batch.full_kspace, (320,320))
        full_kspace = batch.full_kspace
        if self.mode == "interpolation":
            masked_kspace = batch.masked_kspace
        else:
            masked_kspace = batch.full_kspace
        # masked_kspace = complex_center_crop(masked_kspace, (320,320))
        mask = batch.mask
        reconstructed_kspace = self(masked_kspace, mask)
        full_mri_image = kspace_to_mri(full_kspace, (320,320))
        undersampled_mri_image = kspace_to_mri(masked_kspace, (320,320))
        reconstructed_mri_image = kspace_to_mri(reconstructed_kspace, (320,320))
        return full_kspace, reconstructed_kspace, full_mri_image, undersampled_mri_image, reconstructed_mri_image
    
    def training_step(self, batch: KspaceUNetSample, batch_idx):
        full_kspace, reconstructed_kspace, full_mri_image, undersampled_mri_image, reconstructed_mri_image = self.shared_step(batch)
        _, teacher_kspace_output, _, _, teacher_mri_output = self.teacher.shared_step(batch)
        loss = self.reconstruction_distillation_loss(reconstructed_mri_image, teacher_mri_output, full_mri_image)
        self.log("train/dist_loss", loss, on_epoch=True, on_step=True, prog_bar=True, batch_size=full_kspace.shape[0], sync_dist=True)
        return {
            "loss": loss, 
            "undersampled_mri_image": undersampled_mri_image,
            "reconstructed_mri_image": reconstructed_mri_image,
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
        recon_loss = F.mse_loss(student_output, target)
        distill_loss = F.mse_loss(student_output, teacher_output)
        return alpha * distill_loss + (1 - alpha) * recon_loss


if __name__ == "__main__":
    teacher =
