from typing import Union, Optional, Tuple, List
import torch
from diffusers import DiffusionPipeline
from pl_modules.ldm_module import LDM
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from modules.transforms import unnorm, norm, KspaceUNetDataTransform
    
def index_of(tensor: torch.Tensor, value: torch.Tensor):
    idx = (tensor == value.item()).nonzero(as_tuple=True)[0]
    return idx.item() if idx.numel() > 0 else -1  # return -1 if not found
    
class ConsistencyGuidanceSampler(DiffusionPipeline):
    def __init__(
            self, ldm: LDM, 
            scheduler: DDIMScheduler | DDPMScheduler, 
            num_inference_steps: int = 50,
            cgs: bool = False,
            guidance_scale: int = 100, 
        ):

        super().__init__()
            
        self.register_modules(
            unet=ldm.model, 
            scheduler=scheduler,  
            first_stage=ldm.first_stage,
        )
        self.scheduler.set_timesteps(1000)
        self.num_inference_steps = num_inference_steps
        self.rescale_latents = ldm.rescale_latents
        self.cgs = cgs
        self.guidance_scale = guidance_scale

    
    @torch.no_grad()
    def __call__(
        self,
        batch: KspaceUNetDataTransform,
        timesteps: int =  None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        **kwargs,
    ) ->  Tuple:
        if timesteps is not None:
            self.num_inference_steps = timesteps

        k = batch.masked_kspace.permute(0, 3, 1, 2).contiguous()
        k, mean, std = norm(k)
        with torch.inference_mode():
            z_hat = self.first_stage.encode(k)[0]

        rescale_factor = 1.0

        if self.rescale_latents:
            rescale_factor = (1.0 / z_hat.flatten().std())

        # set step values
        z_hat = rescale_factor * z_hat
        noise = torch.randn_like(z_hat)
        t = torch.tensor([self.num_inference_steps])
        z_hat = self.scheduler.add_noise(z_hat, noise, t).contiguous()
        
        for t in self.progress_bar(self.scheduler.timesteps[index_of(self.scheduler.timesteps,t):]):
            # 1. predict noise model_output
            with torch.inference_mode():
                model_output = self.unet(z_hat, t).sample

            # 2. compute previous image: x_t -> x_t-1
            z_hat = self.scheduler.step(model_output, t, z_hat, generator=generator).prev_sample

            if self.cgs:
                if t > 80:
                    if t % 20 == 0:
                        z_hat = z_hat / rescale_factor 

                        with torch.inference_mode():
                            k_hat = self.first_stage.decode(z_hat)

                        k_hat = unnorm(k_hat, mean, std).permute(0, 2, 3, 1).contiguous()

                        zero = torch.zeros(1, 1, 1, 1).to(k_hat)
                        
                        # plt.imshow(kspace_to_mri(k_hat).squeeze(0), cmap="gray")
                        # plt.show()

                        k_hat = k_hat - torch.where(batch.mask, k_hat - batch.masked_kspace, zero)

                        # plt.imshow(kspace_to_mri(k_hat).squeeze(0), cmap="gray")
                        # plt.show()

                        k_hat, mean, std = norm(k_hat.permute(0, 3, 1, 2).contiguous())

                        with torch.inference_mode():
                            z_hat = self.first_stage.encode(k_hat)[0]

                        z_hat = z_hat * rescale_factor
                elif t > 20:
                    with torch.enable_grad():
                        pred_x0 = z_hat.clone().detach()
                        pred_x0.requires_grad_(True)

                        pred_x0 = pred_x0 / rescale_factor
                        k_hat = unnorm(self.first_stage.decode(pred_x0), mean, std).permute(0,2,3,1).contiguous()
                        zero  = torch.zeros_like(k_hat)
                        soft_dc = torch.where(batch.mask, k_hat - batch.masked_kspace, zero)
                        loss = soft_dc.pow(2).sum()
                        # print(loss)
                        g = -torch.autograd.grad(loss, pred_x0)[0]*self.guidance_scale
                    with torch.no_grad():
                        z_hat = (z_hat + g).detach()


        # image = (image / 2 + 0.5).clamp(0, 1)
        z_hat = z_hat / rescale_factor
        # z_hat = unnorm(z_hat, mean_z, std_z)
        with torch.inference_mode():
            kspace = self.first_stage.decode(z_hat)
        kspace = unnorm(kspace, mean, std)
        kspace = kspace.permute(0, 2, 3, 1).contiguous()
        return kspace