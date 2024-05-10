# import torch, math
# import torch.nn as nn

# # Function to generate rank of the decomposition.
# def generate_rank(x, y):
#     #return max(0, min(x, y) // 32)
#     return min(min(x, y), 8)

# class LowRankLinear(nn.Module):
#     def __init__(self, in_shape: int, out_shape: int,
#                  base, bias : torch.Tensor, scaling : int = -1, rank : int = -1):
#         super().__init__()
#         """
#         @param in_shape, out_shape : Layer dimensions as per nn.Linear
#         @param rank : Rank of the decomposition. 
#             (if rank is -1, we use 'min(in_shape, out_shape)//2' as our rank instead.)
#         @param base : Initial base weight of the layer (W), kept frozen during training.
#         @param bias : Initial bias of the layer, trainable.
#         @param scaling : Scaling factor of the LoRA decomposition.

#         Representation of linear layer with weight (W_new), where:

#         W_new = W + A @ B

#         Such that A and B are trainable low-rank matrices initialised as uniform and zero initially.
#         """

#         # Generate rank if not provided.
#         if rank == -1:
#             rank = generate_rank(in_shape, out_shape)

#         # Initialise A and B as trainable parameters.
#         alpha_t = torch.empty((out_shape, rank), dtype = torch.float32, requires_grad = True)
#         beta_t = torch.empty((rank, in_shape), dtype = torch.float32, requires_grad = True)

#         # Initialise A and B as uniform and zero.
#         self.alpha = nn.Parameter(alpha_t, requires_grad = True)
#         self.beta = nn.Parameter(beta_t, requires_grad = True)

#         # Initialise bias.
#         self.bias = nn.Parameter(bias.clone(), requires_grad = True)

#         # Initialise A and B with kaiming uniform and zeros.
#         torch.nn.init.kaiming_uniform_(self.alpha, a =  math.sqrt(5))
#         torch.nn.init.zeros_(self.beta)

#         # Initialise base weight and scaling factor.
#         self.base = base.clone()
#         self.base.requires_grad = False

#         # Set scaling factor.
#         if scaling == -1:
#             self.scaling = 0.5
#         else:
#             self.scaling = scaling


#     # Forward pass of the layer.
#     def forward(self, x):
#         # Compute the output of the layer.
#         h = x @ self.base.T + self.scaling * (x @ (self.alpha @ self.beta).T)
#         return h + self.bias

import torch, math
import torch.nn as nn

# Function to generate rank of the decomposition.
def generate_rank(x, y):
    #return max(0, min(x, y) // 32)
    return min(min(x, y), 8)

class LowRankLinear(nn.Module):
    def __init__(self, in_shape: int, out_shape: int,
                 base, bias: torch.Tensor, scaling: int = -1, rank: int = -1, device=None):
        super().__init__()
        """
        @param in_shape, out_shape : Layer dimensions as per nn.Linear
        @param rank : Rank of the decomposition.
        @param base : Initial base weight of the layer (W), kept frozen during training.
        @param bias : Initial bias of the layer, trainable.
        @param scaling : Scaling factor of the LoRA decomposition.
        @param device : Device to which all parameters will be sent ('cuda' or 'cpu')
        """

        # Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() and device is None else device)

        # Generate rank if not provided.
        if rank == -1:
            rank = generate_rank(in_shape, out_shape)

        # Initialise A and B as trainable parameters.
        alpha_t = torch.empty((out_shape, rank), dtype=torch.float32, requires_grad=True).to(self.device)
        beta_t = torch.empty((rank, in_shape), dtype=torch.float32, requires_grad=True).to(self.device)

        # Initialise A and B as uniform and zero.
        self.alpha = nn.Parameter(alpha_t, requires_grad=True)
        self.beta = nn.Parameter(beta_t, requires_grad=True)

        # Initialise bias.
        self.bias = nn.Parameter(bias.clone().to(self.device), requires_grad=True)

        # Initialise A and B with kaiming uniform and zeros.
        torch.nn.init.kaiming_uniform_(self.alpha, a=math.sqrt(5))
        torch.nn.init.zeros_(self.beta)

        # Initialise base weight and scaling factor.
        self.base = base.clone().to(self.device)
        self.base.requires_grad = False

        # Set scaling factor.
        if scaling == -1:
            self.scaling = 0.5
        else:
            self.scaling = scaling

    # Forward pass of the layer.
    def forward(self, x):
        # Compute the output of the layer.
        h = x @ self.base.T + self.scaling * (x @ (self.alpha @ self.beta).T)
        return h + self.bias
