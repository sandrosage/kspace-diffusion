from pl_modules.mri_module import MRIModule
from diffusers.models.autoencoders.vae import Decoder, Encoder
from diffusers.models.autoencoders import AutoencoderKL
from torch.nn import L1Loss
import torch
from modules.transforms import KspaceUNetSample, norm, unnorm, kspace_to_mri
from modules.losses import LPIPSWithDiscriminator, SSIMAndLPIPSWithDiscriminator
from torch import optim
from modules.utils import create_channels
from fastmri.losses import SSIMLoss

class WeightedSSIMKspaceAutoencoder(MRIModule):
    def __init__(self, 
                 in_channels: int = 2, 
                 out_channels: int = 2, 
                 latent_dim: int = 16,
                 n_mult: list[int] = [2, 4, 8, 16],
                 n_channels: int = 32,   
                 num_log_images = 32,
                 ssim_weight: float = 0.7):
        super().__init__(num_log_images)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.n_mult = n_mult
        self.down_layers = len(n_mult) + 1
        self.n_channels = n_channels
        self.ssim_weight = ssim_weight

        self.save_hyperparameters()
        test_input = torch.randn(1, 2, 640, 384)
        down_block_out_channels = create_channels(self.n_mult, chans=self.n_channels)
        up_block_out_channels = create_channels(self.n_mult, chans=self.n_channels)[::-1]
        up_block_out_channels = down_block_out_channels[::-1]
        down_blocks = self.down_layers*("DownEncoderBlock2D", )
        up_blocks = self.down_layers* ("UpDecoderBlock2D", )

        self.encoder = Encoder(
            in_channels=self.in_channels, 
            out_channels=self.latent_dim, 
            down_block_types=down_blocks, 
            block_out_channels=down_block_out_channels,
            double_z=False)
        
        self.decoder = Decoder(
            in_channels=self.latent_dim,
            out_channels=self.out_channels,
            up_block_types=up_blocks,
            block_out_channels=up_block_out_channels
        )

        self.criterion = L1Loss()
        self.criterion_2 = SSIMLoss()

        print("Channels: ", down_block_out_channels)
        print("Z shape: ", self.encode(test_input).shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def shared_step(self, batch: KspaceUNetSample):
        input = batch.full_kspace
        input = input.permute(0,3,1,2).contiguous()
        input, mean, std = norm(input)
        output = self(input)
        loss = self.criterion(input,output)
        output = unnorm(output, mean, std)
        output = output.permute(0,2,3,1).contiguous()
        return loss, output
    
    def training_step(self, batch: KspaceUNetSample, batch_idx):
        loss, output = self.shared_step(batch)

        output_img = kspace_to_mri(output)

        img_loss = self.criterion(output_img, batch.target)

        self.log_dict({
            "train/kspace_l1": loss, 
            "train/img_l1": img_loss
        }, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True, batch_size=batch.full_kspace.shape[0])

        loss += self.ssim_weight*self.criterion_2(output_img, batch.target, batch.max_value)

        self.log("train/total_loss", loss,  on_epoch=True, on_step=True, sync_dist=True, prog_bar=True, batch_size=batch.full_kspace.shape[0])
        return {
            "loss": loss,
            "reconstructions": output_img, 
        }

    def validation_step(self, batch:KspaceUNetSample, batch_idx):
        loss, output = self.shared_step(batch)

        output_img = kspace_to_mri(output)

        img_loss = self.criterion(batch.target, output_img)

        self.log_dict({
            "val/kspace_l1": loss, 
            "val/img_l1": img_loss
        }, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True, batch_size=batch.full_kspace.shape[0])
        
        return {
            "reconstructions": output_img, 
        }
    
    def test_step(self, batch:KspaceUNetSample, batch_idx):
        return_dict = {
            "reconstructions": None,
            "inputs": None,
            "outputs": None
        }

        input = batch.full_kspace
        input = input.permute(0,3,1,2).contiguous()
        input, mean, std = norm(input)
        output = self(input)
        return_dict["outputs"] = output
        output = unnorm(output, mean, std)
        output = output.permute(0,2,3,1).contiguous()
        output_img = kspace_to_mri(output)
        return_dict["inputs"] = input
        return_dict["reconstructions"] = output_img

        return return_dict
    
    def configure_optimizers(self):
        return optim.Adam(list(self.encoder.parameters())+ list(self.decoder.parameters()), lr=1e-4)

class KspaceAutoencoder(MRIModule):
    def __init__(self, 
                 in_channels: int = 2, 
                 out_channels: int = 2, 
                 latent_dim: int = 16,
                 n_mult: list[int] = [2, 4, 8, 16],
                 n_channels: int = 32,   
                 num_log_images = 32):
        super().__init__(num_log_images)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.n_mult = n_mult
        self.down_layers = len(n_mult) + 1
        self.n_channels = n_channels

        self.save_hyperparameters()
        test_input = torch.randn(1, 2, 640, 384)
        down_block_out_channels = create_channels(self.n_mult, chans=self.n_channels)
        up_block_out_channels = create_channels(self.n_mult, chans=self.n_channels)[::-1]
        up_block_out_channels = down_block_out_channels[::-1]
        down_blocks = self.down_layers*("DownEncoderBlock2D", )
        up_blocks = self.down_layers* ("UpDecoderBlock2D", )

        self.encoder = Encoder(
            in_channels=self.in_channels, 
            out_channels=self.latent_dim, 
            down_block_types=down_blocks, 
            block_out_channels=down_block_out_channels,
            double_z=False)
        
        self.decoder = Decoder(
            in_channels=self.latent_dim,
            out_channels=self.out_channels,
            up_block_types=up_blocks,
            block_out_channels=up_block_out_channels
        )

        self.criterion = L1Loss()

        print("Channels: ", down_block_out_channels)
        print("Z shape: ", self.encode(test_input).shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def shared_step(self, batch: KspaceUNetSample):
        input = batch.full_kspace
        input = input.permute(0,3,1,2).contiguous()
        input, mean, std = norm(input)
        output = self(input)
        loss = self.criterion(input,output)
        output = unnorm(output, mean, std)
        output = output.permute(0,2,3,1).contiguous()
        return loss, output
    
    def training_step(self, batch: KspaceUNetSample, batch_idx):
        loss, output = self.shared_step(batch)

        output_img = kspace_to_mri(output)

        img_loss = self.criterion(output_img, batch.target)

        self.log_dict({
            "train/kspace_l1": loss, 
            "train/img_l1": img_loss
        }, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True, batch_size=batch.full_kspace.shape[0])

        return {
            "loss": loss,
            "reconstructions": output_img, 
        }

    def validation_step(self, batch:KspaceUNetSample, batch_idx):
        loss, output = self.shared_step(batch)

        output_img = kspace_to_mri(output)

        img_loss = self.criterion(batch.target, output_img)

        self.log_dict({
            "val/kspace_l1": loss, 
            "val/img_l1": img_loss
        }, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True, batch_size=batch.full_kspace.shape[0])
        
        return {
            "reconstructions": output_img, 
        }
    
    def test_step(self, batch:KspaceUNetSample, batch_idx):
        return_dict = {
            "reconstructions": None,
            "inputs": None,
            "outputs": None
        }

        input = batch.full_kspace
        input = input.permute(0,3,1,2).contiguous()
        input, mean, std = norm(input)
        output = self(input)
        return_dict["outputs"] = output
        output = unnorm(output, mean, std)
        output = output.permute(0,2,3,1).contiguous()
        output_img = kspace_to_mri(output)
        return_dict["inputs"] = input
        return_dict["reconstructions"] = output_img

        return return_dict
        
    def configure_optimizers(self):
        return optim.Adam(list(self.encoder.parameters())+ list(self.decoder.parameters()), lr=1e-4)

class WeightedSSIMKspaceAutoencoderKL(MRIModule):
    def __init__(self,
                in_channels: int = 2, 
                out_channels: int = 2, 
                latent_dim: int = 16,
                n_mult: list[int] = [2, 4, 8, 16],
                n_channels: int = 32,
                sample_posterior: bool = False,   
                disc_start: int = 5000,
                ssim_weight: float = 0.7, 
                perceptual_weight: float = 0.3,  
                kl_weight: float =  0.000001, 
                disc_weight: float = 0.5,
                num_log_images: int = 32):
        super().__init__(num_log_images)

        self.automatic_optimization = False
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.n_mult = n_mult
        self.down_layers = len(n_mult) + 1
        self.n_channels = n_channels
        self.sample_posterior = sample_posterior
        self.disc_start = disc_start
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.kl_weight = kl_weight
        self.disc_weight = disc_weight

        self.save_hyperparameters()

        test_input = torch.randn(1, 2, 640, 384)
        block_out_channels = create_channels(self.n_mult, chans=self.n_channels)
        down_blocks = self.down_layers*("DownEncoderBlock2D", )
        up_blocks = self.down_layers* ("UpDecoderBlock2D", )
        
        print(f"Start global step for discriminator: {self.disc_start}")
        print(f"Weights for Loss: P: {self.perceptual_weight} | S: {self.ssim_weight} | KL: {self.kl_weight} | D: {self.disc_weight}")

        self.loss = SSIMAndLPIPSWithDiscriminator(
            disc_start=disc_start, 
            disc_in_channels=self.in_channels, 
            perceptual_weight=self.perceptual_weight, 
            ssim_weight=self.ssim_weight,
            kl_weight=kl_weight,
            disc_weight=self.disc_weight)
        
        self.criterion = L1Loss()

        self.model = AutoencoderKL(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            down_block_types=down_blocks, 
            up_block_types=up_blocks, 
            block_out_channels=block_out_channels, 
            latent_channels=self.latent_dim,
            layers_per_block=2)
        
        print("Channels: ", block_out_channels)
        print("Z shape: ", self.encode(test_input)[0].shape)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        posterior = self.model.encode(x).latent_dist # posterior: DiagonalGaussianDistribution
        if self.sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        return z, posterior
    

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z).sample
    
    def forward(self, x: torch.Tensor, sample_posterior: bool = False) -> torch.Tensor:
        return self.model.forward(x, sample_posterior).sample
    
    def __forward(self, x: torch.Tensor):
        z, posterior = self.encode(x)
        dec = self.decode(z)
        return dec, posterior
    
    def training_step(self, batch: KspaceUNetSample, batch_idx):
        # 1. Preprocessing of batch
        input = batch.full_kspace
        input = input.permute(0,3,1,2).contiguous()
        input, mean, std = norm(input)

        # 2. Call model forward
        output, posterior = self.__forward(input)

        # 3. Loss calculation
        opt_0, opt_1 = self.optimizers()

        # 3.1 Loss of generator
        aeloss, log_dict_ae = self.loss(input, output, posterior, 0, self.global_step, mean, std, batch.max_value, batch.target,  
                                        last_layer=self.__get_last_layer(), split="train")
        
        opt_0.zero_grad()
        self.manual_backward(aeloss)
        opt_0.step()

        # 3.2 Loss the discriminator
        discloss, log_dict_disc = self.loss(input, output, posterior, 1, self.global_step, mean, std,batch.max_value, batch.target,  
                                            last_layer=self.__get_last_layer(), split="train")

        opt_1.zero_grad()
        self.manual_backward(discloss)
        opt_1.step()

        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=input.shape[0], sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=input.shape[0], sync_dist=True)

        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=input.shape[0], sync_dist=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=input.shape[0], sync_dist=True)

        output = unnorm(output, mean, std).permute(0, 2, 3, 1).contiguous()
        output_img = kspace_to_mri(output)

        img_loss = self.criterion(batch.target, output_img)
        self.log("train/img_l1", img_loss, on_epoch=True, on_step=True, sync_dist=True, batch_size=input.shape[0])

        return {
            "reconstructions": output_img
        }
    
    def validation_step(self, batch: KspaceUNetSample, batch_idx):
        # 1. Preprocessing of batch
        input = batch.full_kspace
        input = input.permute(0,3,1,2).contiguous()
        input, mean, std = norm(input)

        # 2. Call model forward
        output, posterior = self.__forward(input)

        # 3.1 Loss of generator
        _, log_dict_ae = self.loss(input, output, posterior, 0, self.global_step, mean, std, batch.max_value, batch.target,  
                                        last_layer=self.__get_last_layer(), split="val")

        # 3.2 Loss the discriminator
        _, log_dict_disc = self.loss(input, output, posterior, 1, self.global_step, mean, std, batch.max_value, batch.target,  
                                            last_layer=self.__get_last_layer(), split="val")
        
        self.log_dict(log_dict_ae, on_epoch=True, on_step=True, sync_dist=True)
        self.log_dict(log_dict_disc, on_epoch=True, on_step=True, sync_dist=True)

        output = unnorm(output, mean, std).permute(0, 2, 3, 1).contiguous()
        output_img = kspace_to_mri(output)

        img_loss = self.criterion(batch.target, output_img)
        self.log("val/img_l1", img_loss, on_epoch=True, on_step=True, sync_dist=True, batch_size=input.shape[0])

        return {
           "reconstructions": output_img
        }
    def test_step(self, batch: KspaceUNetSample, batch_idx):

        return_dict = {
            "reconstructions": None,
            "inputs": None,
            "outputs": None
        }

        input = batch.full_kspace
        input = input.permute(0,3,1,2).contiguous()
        input, mean, std = norm(input)
        output, _ = self.__forward(input)
        return_dict["outputs"] = output
        output = unnorm(output, mean, std).permute(0, 2, 3, 1).contiguous()
        output_img = kspace_to_mri(output)
        return_dict["inputs"] = input
        return_dict["reconstructions"] = output_img
        return return_dict

    def configure_optimizers(self):
        lr = 1e-5 # LDPM Paper reference

        opt_ae = torch.optim.Adam(
            list(self.model.parameters()),
            lr=lr, 
            betas=(0.5, 0.9))
        
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr, betas=(0.5, 0.9))
        
        return [opt_ae, opt_disc], []
    
    def __get_last_layer(self):
        return self.model.decoder.conv_out.weight



class KspaceAutoencoderKL(MRIModule):
    def __init__(self,
                in_channels: int = 2, 
                out_channels: int = 2, 
                latent_dim: int = 16,
                n_mult: list[int] = [2, 4, 8, 16],
                n_channels: int = 32,
                sample_posterior: bool = False,    
                num_log_images: int = 32,  
                disc_start: int = 5000, 
                perceptual_weight: float = 1.0,  
                kl_weight: float =  0.000001, 
                disc_weight: float = 0.5):
        super().__init__(num_log_images)

        self.automatic_optimization = False
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.n_mult = n_mult
        self.down_layers = len(n_mult) + 1
        self.n_channels = n_channels
        self.sample_posterior = sample_posterior
        self.disc_start = disc_start
        self.perceptual_weight = perceptual_weight
        self.kl_weight = kl_weight
        self.disc_weight = disc_weight

        self.save_hyperparameters()

        test_input = torch.randn(1, 2, 640, 384)
        block_out_channels = create_channels(self.n_mult, chans=self.n_channels)
        down_blocks = self.down_layers*("DownEncoderBlock2D", )
        up_blocks = self.down_layers* ("UpDecoderBlock2D", )

        self.loss = LPIPSWithDiscriminator(
            disc_start=self.disc_start,
            kl_weight=self.kl_weight, 
            disc_weight=self.disc_weight,
            perceptual_weight=self.perceptual_weight,
            disc_in_channels=self.in_channels)
        
        self.criterion = L1Loss()

        self.model = AutoencoderKL(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            down_block_types=down_blocks, 
            up_block_types=up_blocks, 
            block_out_channels=block_out_channels, 
            latent_channels=self.latent_dim,
            layers_per_block=2)
        
        print("Channels: ", block_out_channels)
        print("Z shape: ", self.encode(test_input)[0].shape)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        posterior = self.model.encode(x).latent_dist # posterior: DiagonalGaussianDistribution
        if self.sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        return z, posterior
    

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z).sample
    
    def forward(self, x: torch.Tensor, sample_posterior: bool = False) -> torch.Tensor:
        return self.model.forward(x, sample_posterior).sample
    
    def __forward(self, x: torch.Tensor):
        z, posterior = self.encode(x)
        dec = self.decode(z)
        return dec, posterior
    
    def training_step(self, batch: KspaceUNetSample, batch_idx):
        # 1. Preprocessing of batch
        input = batch.full_kspace
        input = input.permute(0,3,1,2).contiguous()
        input, mean, std = norm(input)

        # 2. Call model forward
        output, posterior = self.__forward(input)

        # 3. Loss calculation
        opt_0, opt_1 = self.optimizers()

        # 3.1 Loss of generator
        aeloss, log_dict_ae = self.loss(input, output, posterior, 0, self.global_step, mean, std, batch.target,  
                                        last_layer=self.__get_last_layer(), split="train")
        
        opt_0.zero_grad()
        self.manual_backward(aeloss)
        opt_0.step()

        # 3.2 Loss the discriminator
        discloss, log_dict_disc = self.loss(input, output, posterior, 1, self.global_step, mean, std, batch.target,  
                                            last_layer=self.__get_last_layer(), split="train")

        opt_1.zero_grad()
        self.manual_backward(discloss)
        opt_1.step()

        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=input.shape[0], sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=input.shape[0], sync_dist=True)

        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=input.shape[0], sync_dist=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=input.shape[0], sync_dist=True)

        output = unnorm(output, mean, std).permute(0, 2, 3, 1).contiguous()
        output_img = kspace_to_mri(output)

        img_loss = self.criterion(batch.target, output_img)
        self.log("train/img_l1", img_loss, on_epoch=True, on_step=True, sync_dist=True, batch_size=input.shape[0])

        return {
            "reconstructions": output_img
        }
    
    def validation_step(self, batch: KspaceUNetSample, batch_idx):
        # 1. Preprocessing of batch
        input = batch.full_kspace
        input = input.permute(0,3,1,2).contiguous()
        input, mean, std = norm(input)

        # 2. Call model forward
        output, posterior = self.__forward(input)

        # 3.1 Loss of generator
        _, log_dict_ae = self.loss(input, output, posterior, 0, self.global_step, mean, std, batch.target,  
                                        last_layer=self.__get_last_layer(), split="val")

        # 3.2 Loss the discriminator
        _, log_dict_disc = self.loss(input, output, posterior, 1, self.global_step, mean, std, batch.target,  
                                            last_layer=self.__get_last_layer(), split="val")
        
        self.log_dict(log_dict_ae, on_epoch=True, on_step=True, sync_dist=True, batch_size=input.shape[0])
        self.log_dict(log_dict_disc, on_epoch=True, on_step=True, sync_dist=True, batch_size=input.shape[0])

        output = unnorm(output, mean, std).permute(0, 2, 3, 1).contiguous()
        output_img = kspace_to_mri(output)

        img_loss = self.criterion(batch.target, output_img)
        self.log("val/img_l1", img_loss, on_epoch=True, on_step=True, sync_dist=True, batch_size=input.shape[0])

        return {
           "reconstructions": output_img
        }
    
    def configure_optimizers(self):
        lr = 1e-5 # LDPM Paper reference

        opt_ae = torch.optim.Adam(
            list(self.model.parameters()),
            lr=lr, 
            betas=(0.5, 0.9))
        
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr, betas=(0.5, 0.9))
        
        return [opt_ae, opt_disc], []
    
    def __get_last_layer(self):
        return self.model.decoder.conv_out.weight

