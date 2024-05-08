import numpy as np
import torch
import torch.nn as nn
import os
from src.compression.LowRankLinear import LowRankLinear


def getBase(model, basepath=""):
    """
    @param model : The original LeNet.
    
    @return The weights and bias needed to act as the base for the 
        low-rank version of the custom linear layers.
    """

    wd = model.state_dict()
    w = [
         wd['blocks.0.mhsa.q_mappings.0.weight'],  wd['blocks.0.mhsa.q_mappings.1.weight'],
         wd['blocks.0.mhsa.k_mappings.0.weight'],  wd['blocks.0.mhsa.k_mappings.1.weight'],
         wd['blocks.0.mhsa.v_mappings.0.weight'],  wd['blocks.0.mhsa.v_mappings.1.weight'],
         wd['blocks.0.mlp.0.weight'], wd['blocks.0.mlp.2.weight'],
         wd['blocks.1.mhsa.q_mappings.0.weight'],  wd['blocks.1.mhsa.q_mappings.1.weight'],
         wd['blocks.1.mhsa.k_mappings.0.weight'],  wd['blocks.1.mhsa.k_mappings.1.weight'],
         wd['blocks.1.mhsa.v_mappings.0.weight'],  wd['blocks.1.mhsa.v_mappings.1.weight'],
         wd['blocks.1.mlp.0.weight'], wd['blocks.1.mlp.2.weight']
        ]
    b = [
        wd['blocks.0.mhsa.q_mappings.0.bias'],  wd['blocks.0.mhsa.q_mappings.1.bias'],
         wd['blocks.0.mhsa.k_mappings.0.bias'],  wd['blocks.0.mhsa.k_mappings.1.bias'],
         wd['blocks.0.mhsa.v_mappings.0.bias'],  wd['blocks.0.mhsa.v_mappings.1.bias'],
         wd['blocks.0.mlp.0.bias'], wd['blocks.0.mlp.2.bias'],
         wd['blocks.1.mhsa.q_mappings.0.bias'],  wd['blocks.1.mhsa.q_mappings.1.bias'],
         wd['blocks.1.mhsa.k_mappings.0.bias'],  wd['blocks.1.mhsa.k_mappings.1.bias'],
         wd['blocks.1.mhsa.v_mappings.0.bias'],  wd['blocks.1.mhsa.v_mappings.1.bias'],
         wd['blocks.1.mlp.0.bias'], wd['blocks.1.mlp.2.bias'] 
        ]

    base_dict = {
        'blocks.0.mhsa.q_mappings.0.weight' : wd['blocks.0.mhsa.q_mappings.0.weight'],
        'blocks.0.mhsa.q_mappings.0.bias' : wd['blocks.0.mhsa.q_mappings.0.bias'],
        'blocks.0.mhsa.q_mappings.1.weight' : wd['blocks.0.mhsa.q_mappings.1.weight'],
        'blocks.0.mhsa.q_mappings.1.bias' : wd['blocks.0.mhsa.q_mappings.1.bias'],
        'blocks.0.mhsa.k_mappings.0.weight' : wd['blocks.0.mhsa.k_mappings.0.weight'],
        'blocks.0.mhsa.k_mappings.0.bias' : wd['blocks.0.mhsa.k_mappings.0.bias'],
        'blocks.0.mhsa.k_mappings.1.weight' : wd['blocks.0.mhsa.k_mappings.1.weight'],
        'blocks.0.mhsa.k_mappings.1.bias' : wd['blocks.0.mhsa.k_mappings.1.bias'],
        'blocks.0.mhsa.v_mappings.0.weight' : wd['blocks.0.mhsa.v_mappings.0.weight'],
        'blocks.0.mhsa.v_mappings.0.bias' : wd['blocks.0.mhsa.v_mappings.0.bias'],
        'blocks.0.mhsa.v_mappings.1.weight' : wd['blocks.0.mhsa.v_mappings.1.weight'],
        'blocks.0.mhsa.v_mappings.1.bias' : wd['blocks.0.mhsa.v_mappings.1.bias'],
        'blocks.0.mlp.0.weight' : wd['blocks.0.mlp.0.weight'],
        'blocks.0.mlp.0.bias' : wd['blocks.0.mlp.0.bias'],
        'blocks.0.mlp.2.weight' : wd['blocks.0.mlp.2.weight'],
        'blocks.0.mlp.2.bias' : wd['blocks.0.mlp.2.bias'],
        'blocks.1.mhsa.q_mappings.0.weight' : wd['blocks.1.mhsa.q_mappings.0.weight'],
        'blocks.1.mhsa.q_mappings.0.bias' : wd['blocks.1.mhsa.q_mappings.0.bias'],
        'blocks.1.mhsa.q_mappings.1.weight' : wd['blocks.1.mhsa.q_mappings.1.weight'],
        'blocks.1.mhsa.q_mappings.1.bias' : wd['blocks.1.mhsa.q_mappings.1.bias'],
        'blocks.1.mhsa.k_mappings.0.weight' : wd['blocks.1.mhsa.k_mappings.0.weight'],
        'blocks.1.mhsa.k_mappings.0.bias' : wd['blocks.1.mhsa.k_mappings.0.bias'],
        'blocks.1.mhsa.k_mappings.1.weight' : wd['blocks.1.mhsa.k_mappings.1.weight'],
        'blocks.1.mhsa.k_mappings.1.bias' : wd['blocks.1.mhsa.k_mappings.1.bias'],
        'blocks.1.mhsa.v_mappings.0.weight' : wd['blocks.1.mhsa.v_mappings.0.weight'],
        'blocks.1.mhsa.v_mappings.0.bias' : wd['blocks.1.mhsa.v_mappings.0.bias'],
        'blocks.1.mhsa.v_mappings.1.weight' : wd['blocks.1.mhsa.v_mappings.1.weight'],
        'blocks.1.mhsa.v_mappings.1.bias' : wd['blocks.1.mhsa.v_mappings.1.bias'],
        'blocks.1.mlp.0.weight' : wd['blocks.1.mlp.0.weight'],
        'blocks.1.mlp.0.bias' : wd['blocks.1.mlp.0.bias'],
        'blocks.1.mlp.2.weight' : wd['blocks.1.mlp.2.weight'],
        'blocks.1.mlp.2.bias' : wd['blocks.1.mlp.2.bias']
    }
    if basepath != "":
        if not os.path.exists(basepath):
            os.makedirs(basepath)
        fp = os.path.join(basepath, "lora_bases.pt")
        torch.save(base_dict, fp)
        
    return w, b



# def getBase(model, basepath=""):
#     """
#     @param model : The original LeNet.

#     @return The weights and bias needed to act as the base for the
#         low-rank version of the custom linear layers.
#     """
#     base_dict = {}
#     wd = model.state_dict()
#     for name, param in wd.items():
#         base_dict[name] = param

#     if basepath != "":
#         if not os.path.exists(basepath):
#             os.makedirs(basepath)
#         fp = os.path.join(basepath, "lora_bases.pt")
#         torch.save(base_dict, fp)

#     return base_dict

def load_sd_decomp(org_sd, model, decomposed_layers):
    """
    @param org_sd : The state_dict when the model is ongoing.
    @param model : The decomp model with decomposed layers.
    @param decomposed_layers : The decomposed layers in decomp model.

    @return The new model with the old state dictionary loaded in.
    """
    new_sd = model.state_dict()
    for k, v in org_sd.items():
        if k not in decomposed_layers:
            new_sd[k] = v
    model.load_state_dict(new_sd)

def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size] # the first dimension is the channel
                # print("patch type is", type(patch))
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

# class MyMSA(nn.Module):
#     def __init__(self, base_dict, d, n_heads=2, rank=-1):
#         super(MyMSA, self).__init__()
#         self.d = d
#         self.n_heads = n_heads

#         assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

#         d_head = int(d / n_heads)
#         self.q_mappings = nn.ModuleList([LowRankLinear(d_head, d_head, base_dict[f'q_mappings.{i}.weight'], base_dict[f'q_mappings.{i}.bias'], rank=rank) for i in range(self.n_heads)])
#         self.k_mappings = nn.ModuleList([LowRankLinear(d_head, d_head, base_dict[f'k_mappings.{i}.weight'], base_dict[f'k_mappings.{i}.bias'], rank=rank) for i in range(self.n_heads)])
#         self.v_mappings = nn.ModuleList([LowRankLinear(d_head, d_head, base_dict[f'v_mappings.{i}.weight'], base_dict[f'v_mappings.{i}.bias'], rank=rank) for i in range(self.n_heads)])
#         self.d_head = d_head
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, sequences):
#         # Sequences has shape (N, seq_length, token_dim)
#         # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
#         # And come back to    (N, seq_length, item_dim)  (through concatenation)
#         result = []
#         for sequence in sequences:
#             seq_result = []
#             for head in range(self.n_heads):
#                 q_mapping = self.q_mappings[head]
#                 k_mapping = self.k_mappings[head]
#                 v_mapping = self.v_mappings[head]

#                 seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
#                 q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

#                 attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
#                 seq_result.append(attention @ v)
#             result.append(torch.hstack(seq_result))
#         return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class MyMSA(nn.Module):
    def __init__(self, weights : list, bias : list, d, cpt, n_heads=2, rank= -1):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([LowRankLinear(d_head, d_head, weights[i+8*cpt], bias[i+8*cpt], rank = rank) for i in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([LowRankLinear(d_head, d_head, weights[i+2+8*cpt], bias[i+2+8*cpt], rank = rank) for i in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([LowRankLinear(d_head, d_head, weights[i+4+8*cpt], bias[i+4+8*cpt], rank = rank) for i in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
    

class MyViTBlock(nn.Module):
    def __init__(self, weights, bias, hidden_d, n_heads, cpt, mlp_ratio=4, rank=-1):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(weights, bias, hidden_d, cpt, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            LowRankLinear(hidden_d, mlp_ratio * hidden_d, weights[6+8*cpt], bias[6+8*cpt], rank = rank),
            nn.GELU(),
            LowRankLinear(mlp_ratio * hidden_d, hidden_d, weights[7+8*cpt], bias[7+8*cpt], rank = rank)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out
    
class ViT_LowRank(nn.Module):
    def __init__(self, weights : list, bias : list, rank, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(ViT_LowRank, self).__init__()
        
        # Attributes
        self.chw = chw # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        
        # 3) Positional embedding
        self.register_buffer('positional_embeddings', get_positional_embeddings(n_patches ** 2 + 1, hidden_d), persistent=False)
        
        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([MyViTBlock(weights, bias, hidden_d, n_heads, cpt, rank=rank) for cpt in range(n_blocks)])
        
        # 5) Classification MLPk
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)
        
        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)
        
        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        
        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)
        
        # Transformer Blocks
        for block in self.blocks:
            out = block(out)
            
        # Getting the classification token only
        out = out[:, 0]
        
        return self.mlp(out) # Map to output dimension, output category distribution