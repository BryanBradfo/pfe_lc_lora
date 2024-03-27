import torch, math
import torch.nn as nn

def generate_rank(x, y):
    #return max(0, min(x, y) // 32)
    return min(min(x, y), 8)

class LowRankLinear(nn.Module):
    def __init__(self, in_shape: int, out_shape: int,
                 base, bias : torch.Tensor, scaling : int = -1, rank : int = -1):
        super().__init__()
        """
        @param in_shape, out_shape : Layer dimensions as per nn.Linear
        @param rank : Rank of the decomposition. 
            (if rank is -1, we use 'min(in_shape, out_shape)//2' as our rank instead.)
        @param base : Initial base weight of the layer (W), kept frozen during training.
        @param bias : Initial bias of the layer, trainable.
        @param scaling : Scaling factor of the LoRA decomposition.

        Representation of linear layer with weight (W_new), where:

        W_new = W + A @ B

        Such that A and B are trainable low-rank matrices initialised as uniform and zero initially.
        """
        if rank == -1:
            rank = generate_rank(in_shape, out_shape)
        alpha_t = torch.empty((out_shape, rank), dtype = torch.float32, requires_grad = True)
        beta_t = torch.empty((rank, in_shape), dtype = torch.float32, requires_grad = True)
        self.alpha = nn.Parameter(alpha_t, requires_grad = True)
        self.beta = nn.Parameter(beta_t, requires_grad = True)
        self.bias = nn.Parameter(bias.clone(), requires_grad = True)
        torch.nn.init.kaiming_uniform_(self.alpha, a =  math.sqrt(5))
        torch.nn.init.zeros_(self.beta)
        self.base = base.clone()
        self.base.requires_grad = False
        if scaling == -1:
            self.scaling = 0.5
        else:
            self.scaling = scaling

    def forward(self, x):
        h = x @ self.base.T + self.scaling * (x @ (self.alpha @ self.beta).T)
        return h + self.bias