from modules.rupali import MultiChannelVAE, VAE
from pl_modules.mri_module import MRIModule

class RupaliAutoencoderModule(MRIModule):
    def __init__(self, 
                 num_log_images = 16, 
                 n_channels = 2,
                 vae_class = VAE,
                 latent_dim = 128,
                 n_feats = [[32, 64, 128, 256], [32, 64, 128, 256]]):
        super().__init__(num_log_images)

        self.model = MultiChannelVAE(n_channels=n_channels, vae_class=vae_class, latent_dim=latent_dim, n_)

