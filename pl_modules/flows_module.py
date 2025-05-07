import normflows as nf
from pl_modules.mri_module import MRIModule

class NormFlowModule(MRIModule):
    def __init__(self, num_log_images = 16):
        super().__init__(num_log_images)
        self.base = nf.distributions.base.DiagGaussian(2)
        # Define list of flows
        num_layers = 32
        self.flows = []
        for i in range(num_layers):
            # Neural network with two hidden layers having 64 units each
            # Last layer is initialized by zeros making training more stable
            param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
            # Add flow layer
            self.flows.append(nf.flows.AffineCouplingBlock(param_map))
            # Swap dimensions
            self.flows.append(nf.flows.Permute(2, mode='swap'))
        self.model = nf.NormalizingFlow(self.base, self.flows)