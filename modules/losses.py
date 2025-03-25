import torch
import torch.nn as nn


class ELBOLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,rec_loss: torch.Tensor, kl_loss: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        elbo = (rec_loss + kl_loss) / len(input) # we want to maximize the elbo -> so we minimize the negative of the elbo
        return - elbo

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
