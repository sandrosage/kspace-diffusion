import torch
from torch import nn
from typing import List
from modules.utils import kspace_to_image
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self,config, in_channels: int,out_channels: int, latent_dim: int, hidden_dims: List[int] = None):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.coils=in_channels
        self.out_channels=out_channels
        self.config= config
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        self.hidden_dims = hidden_dims
        for i, h_dim in enumerate(hidden_dims):
            self.add_module(f'encoder_{i}_0', nn.Conv3d(in_channels=in_channels, out_channels=h_dim,
                                                        kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)))
            self.add_module(f'encoder_{i}_1', nn.BatchNorm3d(h_dim))
            self.add_module(f'encoder_{i}_2', nn.LeakyReLU())
            in_channels = h_dim  # Update in_channels for the next layer


        # conv_output_shape = self._calculate_conv_output_shape(input_shape=(8, 2, self.config.training.out_coils, 320, 320))
        conv_output_shape = self._calculate_conv_output_shape(input_shape=(1, 2, 320, 320))

        # Latent space representation
        self.W_mu = nn.Linear(conv_output_shape, latent_dim)
        self.W_var = nn.Linear(conv_output_shape, latent_dim)

        # Build Decoder
        hidden_dims.reverse()
        self.decoder_input = nn.Linear(latent_dim, conv_output_shape)

        for i in range(len(hidden_dims) - 1):
            self.add_module(f'decoder_{i}_0', nn.ConvTranspose3d(hidden_dims[i], hidden_dims[i + 1],
                                                                 kernel_size=(3, 3, 3), stride=(1, 2, 2),
                                                                 padding=(1, 1, 1), output_padding=(0, 1, 1)))
            self.add_module(f'decoder_{i}_1', nn.BatchNorm3d(hidden_dims[i + 1]))
            self.add_module(f'decoder_{i}_2', nn.LeakyReLU())

        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(hidden_dims[-1], hidden_dims[-1],
                               kernel_size=(3, 3, 3), stride=(1, 2, 2),
                               padding=(1, 1, 1), output_padding=(0, 1, 1)),
            nn.BatchNorm3d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv3d(hidden_dims[-1], out_channels=self.out_channels,
                      kernel_size=(3, 3, 3), padding=1),
            nn.Tanh())

    def _calculate_conv_output_shape(self, input_shape):
        """Helper function to calculate the output shape after all convolutional layers"""
        batch_size, channels, coils, height, width = input_shape
        x = torch.rand(input_shape)
        for i in range(len(self.hidden_dims)):
            x = getattr(self, f'encoder_{i}_0')(x)
        flattened_size = x.numel() // batch_size  # Flattened size for a single batch element
        return flattened_size
    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        for i in range(len(self.hidden_dims)):
            input = getattr(self, f'encoder_{i}_0')(input)
            input = getattr(self, f'encoder_{i}_1')(input)
            input = getattr(self, f'encoder_{i}_2')(input)
        result = torch.flatten(input, start_dim=1)
        mu = self.W_mu(result)
        log_var = self.W_var(result)
        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1,  self.hidden_dims[0], 10, 10)  # Adjust based on the encoder's final output shape
        for i in range(len(self.hidden_dims) - 1):
            result = getattr(self, f'decoder_{i}_0')(result)
            result = getattr(self, f'decoder_{i}_1')(result)
            result = getattr(self, f'decoder_{i}_2')(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        mu, log_var = self.encode(input)
        # log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        # print("recon",torch.min(recons), torch.max(recons))
        # print("input", torch.min(input), torch.max(input))
        predicted_image = kspace_to_image(recons)
        true_image = kspace_to_image(input)



        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        kspace_loss = F.mse_loss(recons, input)
        # print(recons.shape)
        # print(input.shape)
        # print("kspace_loss",kspace_loss)
        recons_loss = F.mse_loss(predicted_image, true_image)



        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss =   self.config.training.kspace_weight * kspace_loss + self.config.training.recon_weight * recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self, num_samples: int, current_device: int) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]



class MultiChannelVAE(nn.Module):
    def __init__(self, n_channels: int, in_channels:int,out_coils:int,vae_class: nn.Module, latent_dim: int, **vae_kwargs):
        super(MultiChannelVAE, self).__init__()
        self.n_channels = n_channels
        self.out_coils = out_coils
        self.latent_dim = latent_dim
        self.vae_class = vae_class
        self.vae_kwargs = vae_kwargs
        # Manually add each VAE as a named module with underscores
        for i in range(n_channels):
            vae = self.vae_class(
                in_channels=in_channels,  # Each channel has 15 feature maps
                out_coils=out_coils,
                latent_dim=latent_dim,
                **vae_kwargs
            )
            self.add_module(f'vaes_{i}', vae)

    # def encode(self, x):
    #     return [self.vae[i].encode(x[i]) for i in range(self.n_channels)]
    #
    # def decode(self, z):
    #     p = []
    #     for i in range(self.n_channels):
    #         pi = [self.vae[i].decode(z[j]) for j in range(self.n_channels)]
    #         p.append(pi)
    #         del pi
    #     return p  # p[x][z]: p(x|z)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for i in range(self.n_channels):
            vae = getattr(self, f'vaes_{i}')
            outputs.append(vae(x[:, i]))
        return torch.stack(outputs, dim=1)





    def loss_function(self, recons: List[torch.Tensor], inputs: torch.Tensor, mus: List[torch.Tensor],
                      log_vars: List[torch.Tensor], **kwargs) -> dict:
        recons_loss = sum(
            self.vaes[ch].loss_function(recons[ch], inputs[:, ch, :, :, :], mus[ch], log_vars[ch], **kwargs)['Reconstruction_Loss']
            for ch in range(self.n_channels)
        )
        kld_loss = sum(
            self.vaes[ch].loss_function(recons[ch], inputs[:, ch, :, :, :], mus[ch], log_vars[ch], **kwargs)['KLD']
            for ch in range(self.n_channels)
        )

        return {'loss': recons_loss + kld_loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

    def sample(self, num_samples: int, current_device: int) -> torch.Tensor:
        # Generate a batch of samples for each VAE channel
        samples = []
        for i in range(self.n_channels):
            vae = getattr(self, f'vaes_{i}')
            z = torch.randn(num_samples, self.latent_dim).to(current_device)
            sample = vae.decode(z)
            samples.append(sample)

        # Stack the samples along the channel dimension
        return torch.stack(samples, dim=1)
