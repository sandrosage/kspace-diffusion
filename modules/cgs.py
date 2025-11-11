from typing import Union, Optional, Tuple, List
import torch
from pl_modules.diffusers_vae_module import WeightedSSIMKspaceAutoencoder, WeightedSSIMKspaceAutoencoderKL
from diffusers import DiffusionPipeline
from pl_modules.ldm_module import LDM
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from modules.transforms import unnorm, norm

class ConsistencyGuidance:
    def __init__(
        self, 
        first_stage: WeightedSSIMKspaceAutoencoderKL | WeightedSSIMKspaceAutoencoder,
        p: int = 200, 
        scale: float = 0.2, 
        ) -> "ConsistencyGuidance":

        self.scale = scale
        self.p = p
        self.first_stage = first_stage
    
    def __call__(
        self, 
        z_hat: torch.Tensor, 
        k: torch.Tensor,  
        mask: torch.Tensor, 
        t: int,
        norm_params: Tuple[torch.Tensor, torch.Tensor],
        ) -> torch.Tensor:

        pred_x0 = z_hat.detach().clone()

        k_hat = self.first_stage.decode(pred_x0)
        k_hat = unnorm(k_hat, norm_params[0], norm_params[1])
        k_hat = k_hat.permute(0, 2, 3, 1).contiguous()

        target_x0 = k.detach().clone()
        mask_x0 = mask.detach().clone()
        zero = torch.zeros(1, 1, 1, 1, 1).to(pred_x0)
        soft_dc = torch.where(mask_x0, k_hat - target_x0, zero)

        if t > self.p:
            # hard guidance: DC
            new_k_hat = k_hat-soft_dc
            new_k_hat = norm(new_k_hat.permute(0, 3, 1, 2).contiguous())
            return self.first_stage.encode(new_k_hat)[0]
        else:
            # soft guidance: consistency
            with torch.enable_grad():
                pred_x0.requires_grad_(True)
                loss = soft_dc.pow(2).sum()
                #loss = (pred_x0 - target_x0).pow(2).mean((1, 2, 3)).sum()
                g = -torch.autograd.grad(loss, pred_x0)[0] * self.scale
            return pred_x0 + g
    
def index_of(tensor: torch.Tensor, value: torch.Tensor):
    idx = (tensor == value.item()).nonzero(as_tuple=True)[0]
    return idx.item() if idx.numel() > 0 else -1  # return -1 if not found
    
class ConsistencyGuidanceSampler(DiffusionPipeline):
    def __init__(
            self, ldm: LDM, 
            scheduler: DDIMScheduler | DDPMScheduler, 
            num_inference_steps: int = 50,
        ):

        super().__init__()
        self.register_modules(
            unet=ldm.model, 
            scheduler=scheduler,  
            first_stage=ldm.first_stage
        )
        self.scheduler.set_timesteps(1000)
        self.num_inference_steps = num_inference_steps
        self.rescale_latents = ldm.rescale_latents

    
    @torch.no_grad()
    def __call__(
        self,
        masked_kspace: torch.Tensor,
        timesteps: int =  None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        **kwargs,
    ) ->  Tuple:
        if timesteps is not None:
            self.num_inference_steps = timesteps

        masked_kspace = masked_kspace.permute(0, 3, 1, 2).contiguous()
        masked_kspace, mean, std = norm(masked_kspace)
        z_hat = self.first_stage.encode(masked_kspace)[0]

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
            model_output = self.unet(z_hat, t).sample

            # 2. compute previous image: x_t -> x_t-1
            z_hat = self.scheduler.step(model_output, t, z_hat, generator=generator).prev_sample

        # image = (image / 2 + 0.5).clamp(0, 1)
        z_hat = z_hat / rescale_factor
        # z_hat = unnorm(z_hat, mean_z, std_z)
        kspace = self.first_stage.decode(z_hat)
        kspace = unnorm(kspace, mean, std)
        kspace = kspace.permute(0, 2, 3, 1).contiguous()
        return kspace
    
          