import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import os
import numpy as np
from src.compression.LowRankLinear_parallel import LowRankLinear_parallel



def getBase(model, basepath=""):
    """
    @param model : The original LeNet.
    
    @return The weights and bias needed to act as the base for the 
        low-rank version of the custom linear layers.
    """

    wd = model.state_dict()
    w = [
         wd['enc.0.msa.q.weight'],  
         wd['enc.0.msa.k.weight'],
         wd['enc.0.msa.v.weight'],


         wd['enc.1.msa.q.weight'],
         wd['enc.1.msa.k.weight'],
         wd['enc.1.msa.v.weight'],


         wd['enc.2.msa.q.weight'],
         wd['enc.2.msa.k.weight'],
         wd['enc.2.msa.v.weight'],


         wd['enc.3.msa.q.weight'],
         wd['enc.3.msa.k.weight'],
         wd['enc.3.msa.v.weight'], 


         wd['enc.4.msa.q.weight'], 
         wd['enc.4.msa.k.weight'], 
         wd['enc.4.msa.v.weight'], 


         wd['enc.5.msa.q.weight'], 
         wd['enc.5.msa.k.weight'], 
         wd['enc.5.msa.v.weight'],


         wd['enc.6.msa.q.weight'],
         wd['enc.6.msa.k.weight'], 
         wd['enc.6.msa.v.weight'],


         wd['enc.7.msa.q.weight'],
         wd['enc.7.msa.k.weight'], 
         wd['enc.7.msa.v.weight'], 


         wd['enc.8.msa.q.weight'], 
         wd['enc.8.msa.k.weight'],
         wd['enc.8.msa.v.weight'],


         wd['enc.9.msa.q.weight'], 
         wd['enc.9.msa.k.weight'],
         wd['enc.9.msa.v.weight'], 


         wd['enc.10.msa.q.weight'],
         wd['enc.10.msa.k.weight'],
         wd['enc.10.msa.v.weight'],


         wd['enc.11.msa.q.weight'],
         wd['enc.11.msa.k.weight'],
         wd['enc.11.msa.v.weight'],
        ]
    b = [
         wd['enc.0.msa.q.bias'],
         wd['enc.0.msa.k.bias'],
         wd['enc.0.msa.v.bias'],


         wd['enc.1.msa.q.bias'],
         wd['enc.1.msa.k.bias'],
         wd['enc.1.msa.v.bias'],



         wd['enc.2.msa.q.bias'],
         wd['enc.2.msa.k.bias'],
         wd['enc.2.msa.v.bias'],


         wd['enc.3.msa.q.bias'],
         wd['enc.3.msa.k.bias'],
         wd['enc.3.msa.v.bias'],



         wd['enc.4.msa.q.bias'],
         wd['enc.4.msa.k.bias'],
         wd['enc.4.msa.v.bias'],



         wd['enc.5.msa.q.bias'],
         wd['enc.5.msa.k.bias'],
         wd['enc.5.msa.v.bias'],



         wd['enc.6.msa.q.bias'],
         wd['enc.6.msa.k.bias'],
         wd['enc.6.msa.v.bias'],



         wd['enc.7.msa.q.bias'],
         wd['enc.7.msa.k.bias'],
         wd['enc.7.msa.v.bias'],



         wd['enc.8.msa.q.bias'],
         wd['enc.8.msa.k.bias'],
         wd['enc.8.msa.v.bias'],


         wd['enc.9.msa.q.bias'],
         wd['enc.9.msa.k.bias'],
         wd['enc.9.msa.v.bias'],





         wd['enc.10.msa.q.bias'],
         wd['enc.10.msa.k.bias'],
         wd['enc.10.msa.v.bias'],





         wd['enc.11.msa.q.bias'],
         wd['enc.11.msa.k.bias'],
         wd['enc.11.msa.v.bias'],
        ]

    base_dict = {
        'enc.0.msa.q.weight' : wd['enc.0.msa.q.weight'],
        'enc.0.msa.q.bias' : wd['enc.0.msa.q.bias'],
        
        'enc.0.msa.k.weight' : wd['enc.0.msa.k.weight'],
        'enc.0.msa.k.bias' : wd['enc.0.msa.k.bias'],
        
        'enc.0.msa.v.weight' : wd['enc.0.msa.v.weight'],
        'enc.0.msa.v.bias' : wd['enc.0.msa.v.bias'],
        
        


        'enc.1.msa.q.weight' : wd['enc.1.msa.q.weight'],
        'enc.1.msa.q.bias' : wd['enc.1.msa.q.bias'],
        
        'enc.1.msa.k.weight' : wd['enc.1.msa.k.weight'],
        'enc.1.msa.k.bias' : wd['enc.1.msa.k.bias'],
        
        'enc.1.msa.v.weight' : wd['enc.1.msa.v.weight'],
        'enc.1.msa.v.bias' : wd['enc.1.msa.v.bias'],
        



        'enc.2.msa.q.weight' : wd['enc.2.msa.q.weight'],
        'enc.2.msa.q.bias' : wd['enc.2.msa.q.bias'],
        
        'enc.2.msa.k.weight' : wd['enc.2.msa.k.weight'],
        'enc.2.msa.k.bias' : wd['enc.2.msa.k.bias'],
        
        'enc.2.msa.v.weight' : wd['enc.2.msa.v.weight'],
        'enc.2.msa.v.bias' : wd['enc.2.msa.v.bias'],
        

        


        'enc.3.msa.q.weight' : wd['enc.3.msa.q.weight'],
        'enc.3.msa.q.bias' : wd['enc.3.msa.q.bias'],
        
        'enc.3.msa.k.weight' : wd['enc.3.msa.k.weight'],
        'enc.3.msa.k.bias' : wd['enc.3.msa.k.bias'],
        
        'enc.3.msa.v.weight' : wd['enc.3.msa.v.weight'],
        'enc.3.msa.v.bias' : wd['enc.3.msa.v.bias'],
        

        


        'enc.4.msa.q.weight' : wd['enc.4.msa.q.weight'],
        'enc.4.msa.q.bias' : wd['enc.4.msa.q.bias'],
        
        'enc.4.msa.k.weight' : wd['enc.4.msa.k.weight'],
        'enc.4.msa.k.bias' : wd['enc.4.msa.k.bias'],
        
        'enc.4.msa.v.weight' : wd['enc.4.msa.v.weight'],
        'enc.4.msa.v.bias' : wd['enc.4.msa.v.bias'],
        



        'enc.5.msa.q.weight' : wd['enc.5.msa.q.weight'],
        'enc.5.msa.q.bias' : wd['enc.5.msa.q.bias'],
        
        'enc.5.msa.k.weight' : wd['enc.5.msa.k.weight'],
        'enc.5.msa.k.bias' : wd['enc.5.msa.k.bias'],
        
        'enc.5.msa.v.weight' : wd['enc.5.msa.v.weight'],
        'enc.5.msa.v.bias' : wd['enc.5.msa.v.bias'],
        

        


        'enc.6.msa.q.weight' : wd['enc.6.msa.q.weight'],
        'enc.6.msa.q.bias' : wd['enc.6.msa.q.bias'],
        
        'enc.6.msa.k.weight' : wd['enc.6.msa.k.weight'],
        'enc.6.msa.k.bias' : wd['enc.6.msa.k.bias'],
        
        'enc.6.msa.v.weight' : wd['enc.6.msa.v.weight'],
        'enc.6.msa.v.bias' : wd['enc.6.msa.v.bias'],
        









        'enc.7.msa.q.weight' : wd['enc.7.msa.q.weight'],
        'enc.7.msa.q.bias' : wd['enc.7.msa.q.bias'],
        
        'enc.7.msa.k.weight' : wd['enc.7.msa.k.weight'],
        'enc.7.msa.k.bias' : wd['enc.7.msa.k.bias'],
        
        'enc.7.msa.v.weight' : wd['enc.7.msa.v.weight'],
        'enc.7.msa.v.bias' : wd['enc.7.msa.v.bias'],
        






        'enc.8.msa.q.weight' : wd['enc.8.msa.q.weight'],
        'enc.8.msa.q.bias' : wd['enc.8.msa.q.bias'],
        
        'enc.8.msa.k.weight' : wd['enc.8.msa.k.weight'],
        'enc.8.msa.k.bias' : wd['enc.8.msa.k.bias'],
        
        'enc.8.msa.v.weight' : wd['enc.8.msa.v.weight'],
        'enc.8.msa.v.bias' : wd['enc.8.msa.v.bias'],
        





        'enc.9.msa.q.weight' : wd['enc.9.msa.q.weight'],
        'enc.9.msa.q.bias' : wd['enc.9.msa.q.bias'],
        
        'enc.9.msa.k.weight' : wd['enc.9.msa.k.weight'],
        'enc.9.msa.k.bias' : wd['enc.9.msa.k.bias'],
        
        'enc.9.msa.v.weight' : wd['enc.9.msa.v.weight'],
        'enc.9.msa.v.bias' : wd['enc.9.msa.v.bias'],
        



        'enc.10.msa.q.weight' : wd['enc.10.msa.q.weight'],
        'enc.10.msa.q.bias' : wd['enc.10.msa.q.bias'],
        
        'enc.10.msa.k.weight' : wd['enc.10.msa.k.weight'],
        'enc.10.msa.k.bias' : wd['enc.10.msa.k.bias'],
        
        'enc.10.msa.v.weight' : wd['enc.10.msa.v.weight'],
        'enc.10.msa.v.bias' : wd['enc.10.msa.v.bias'],




        'enc.11.msa.q.weight' : wd['enc.11.msa.q.weight'],
        'enc.11.msa.q.bias' : wd['enc.11.msa.q.bias'],
        
        'enc.11.msa.k.weight' : wd['enc.11.msa.k.weight'],
        'enc.11.msa.k.bias' : wd['enc.11.msa.k.bias'],
        
        'enc.11.msa.v.weight' : wd['enc.11.msa.v.weight'],
        'enc.11.msa.v.bias' : wd['enc.11.msa.v.bias'],
    }
    if basepath != "":
        if not os.path.exists(basepath):
            os.makedirs(basepath)
        fp = os.path.join(basepath, "lora_bases.pt")
        torch.save(base_dict, fp)
        
    # print("w should have ", 3*12, " elements")

    # print("w[0] ('enc.0.msa.q.weight'):",w[0] == wd["enc.0.msa.q.weight"])
    # print("w[1] ('enc.0.msa.k.weight'):",w[1] == wd["enc.0.msa.k.weight"])
    # print("w[2] ('enc.0.msa.v.weight'):",w[2] == wd["enc.0.msa.v.weight"])
    # print("w[3] ('enc.1.msa.q.weight'):",w[3] == wd["enc.1.msa.q.weight"])
    # print("w[4] ('enc.1.msa.k.weight'):",w[4] == wd["enc.1.msa.k.weight"])
    # print("w[5] ('enc.1.msa.v.weight'):",w[5] == wd["enc.1.msa.v.weight"])
    # print("w[6] ('enc.2.msa.q.weight'):",w[6] == wd["enc.2.msa.q.weight"])
    # print("w[7] ('enc.2.msa.k.weight'):",w[7] == wd["enc.2.msa.k.weight"])
    # print("w[8] ('enc.2.msa.v.weight'):",w[8] == wd["enc.2.msa.v.weight"])
    # print(w[8])
    # print(wd["enc.2.msa.v.weight"])
    # print("====================================")
    # print(w[9])
    # print(wd["enc.3.msa.q.weight"])
    # print("w[9] ('enc.3.msa.q.weight'):",w[9] == wd["enc.3.msa.q.weight"])
    # print("w[10] ('enc.3.msa.k.weight'):",w[10] == wd["enc.3.msa.k.weight"])
    # print("w[11] ('enc.3.msa.v.weight'):",w[11] == wd["enc.3.msa.v.weight"])
    # print("w[12] ('enc.4.msa.q.weight'):",w[12] == wd["enc.4.msa.q.weight"])
    # print("w[13] ('enc.4.msa.k.weight'):",w[13] == wd["enc.4.msa.k.weight"])
    # print("w[14] ('enc.4.msa.v.weight'):",w[14] == wd["enc.4.msa.v.weight"])
    # print("w[15] ('enc.5.msa.q.weight'):",w[15] == wd["enc.5.msa.q.weight"])
    # print("w[16] ('enc.5.msa.k.weight'):",w[16] == wd["enc.5.msa.k.weight"])
    # print("w[17] ('enc.5.msa.v.weight'):",w[17] == wd["enc.5.msa.v.weight"])
    # print("w[18] ('enc.6.msa.q.weight'):",w[18] == wd["enc.6.msa.q.weight"])
    # print("w[19] ('enc.6.msa.k.weight'):",w[19] == wd["enc.6.msa.k.weight"])
    # print("w[20] ('enc.6.msa.v.weight'):",w[20] == wd["enc.6.msa.v.weight"])
    # print("w[21] ('enc.7.msa.q.weight'):",w[21] == wd["enc.7.msa.q.weight"])
    # print("w[22] ('enc.7.msa.k.weight'):",w[22] == wd["enc.7.msa.k.weight"])
    # print("w[23] ('enc.7.msa.v.weight'):",w[23] == wd["enc.7.msa.v.weight"])
    # print("w[24] ('enc.8.msa.q.weight'):",w[24] == wd["enc.8.msa.q.weight"])
    # print("w[25] ('enc.8.msa.k.weight'):",w[25] == wd["enc.8.msa.k.weight"])
    # print("w[26] ('enc.8.msa.v.weight'):",w[26] == wd["enc.8.msa.v.weight"])
    # print("w[27] ('enc.9.msa.q.weight'):",w[27] == wd["enc.9.msa.q.weight"])
    # print("w[28] ('enc.9.msa.k.weight'):",w[28] == wd["enc.9.msa.k.weight"])
    # print("w[29] ('enc.9.msa.v.weight'):",w[29] == wd["enc.9.msa.v.weight"])
    # print("w[30] ('enc.10.msa.q.weight'):",w[30] == wd["enc.10.msa.q.weight"])
    # print("w[31] ('enc.10.msa.k.weight'):",w[31] == wd["enc.10.msa.k.weight"])
    # print("w[32] ('enc.10.msa.v.weight'):",w[32] == wd["enc.10.msa.v.weight"])
    # print("w[33] ('enc.11.msa.q.weight'):",w[33] == wd["enc.11.msa.q.weight"])
    # print("w[34] ('enc.11.msa.k.weight'):",w[34] == wd["enc.11.msa.k.weight"])
    # print("w[35] ('enc.11.msa.v.weight'):",w[35] == wd["enc.11.msa.v.weight"])


    # print("The number of elements in w is:", len(w))
    # print("The number of elements in b is:", len(b))

    # print("The number of elements in w is:", len(w))
    # print(b[0])
    return w, b


def load_sd_decomp(org_sd, model, decomposed_layers):
    """
    @param org_sd : The state_dict when the model is ongoing.
    @param model : The decomp model with decomposed layers.
    @param decomposed_layers : The decomposed layers in decomp model.

    @return The new model with the old state dictionary loaded in.
    """
    new_sd = model.state_dict()
    # print("checkpoint state dictionary:",new_sd)
    for k, v in org_sd.items():
        if k not in decomposed_layers:
            new_sd[k] = v
            # print("k:",k)
            # print("v.size:",v.size())
            # print("v:",v)
    # print(new_sd)
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

######################################

class TransformerEncoder(nn.Module):
    def __init__(self, weights, bias, layer, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, weights, bias, layer, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # print("x.size():",x.size())
        # print("self.msa(self.la1(x)).size():",(self.msa(self.la1(x))).size())
        # print("(self.la1(x)).size():",(self.la1(x)).size())
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats: int, weights: list, bias: list, layer: int, head: int = 8, dropout: float = 0., rank: int = -1):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats ** 0.5
        self.d_head = feats // head

        self.q = LowRankLinear_parallel(feats, feats, weights[3*layer+0], bias[3*layer+0], rank=rank)
        self.k = LowRankLinear_parallel(feats, feats, weights[3*layer+1], bias[3*layer+1], rank=rank)
        self.v = LowRankLinear_parallel(feats, feats, weights[3*layer+2], bias[3*layer+2], rank=rank)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        
        q = self.q(x).view(b, n, self.head, self.d_head).transpose(1, 2)
        k = self.k(x).view(b, n, self.head, self.d_head).transpose(1, 2)
        v = self.v(x).view(b, n, self.head, self.d_head).transpose(1, 2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k) / self.sqrt_d, dim=-1)
        attn = torch.einsum("bhij, bhjf->bhif", score, v)
        # print("attn.size():",attn.size())
        # print("attn.flatten(2).size():",attn.flatten(2).size())
        # print("(self.o(attn.flatten(2)).T.size():",self.o((attn.flatten(2)).T).size())
        # print("self.o(attn.flatten(2)).size():",self.o((attn.flatten(2))).size())
        # print("attn.transpose(1, 2).contiguous().view(b, n, self.feats)",attn.transpose(1, 2).contiguous().view(b, n, self.feats))
        attn = attn.transpose(1, 2).contiguous().view(b, n, self.feats)
        o = self.dropout(self.o(attn))
        # o = self.dropout(self.o((attn.flatten(2)).T))
        return o

##############################################

        # positions_q = [i for i, item in enumerate(weights) if 'msa.q.weight' in str(item)]
        # weights_q = [weights[3*i] for i in range(12)]
        # # positions_k = [i for i, item in enumerate(weights) if 'msa.k.weight' in str(item)]
        # weights_k = [weights[3*i+1] for i in range(12)]
        # # positions_v = [i for i, item in enumerate(weights) if 'msa.v.weight' in str(item)]
        # weights_v = [weights[3*i+2] for i in range(12)]

        # # position_bias_q = [i for i, item in enumerate(bias) if 'msa.q.bias' in str(item)]
        # bias_q = torch.tensor([bias[3*i].item() for i in range(12)])
        # # position_bias_k = [i for i, item in enumerate(bias) if 'msa.k.bias' in str(item)]
        # bias_k = torch.tensor([bias[3*i+1].item() for i in range(12)])
        # # position_bias_v = [i for i, item in enumerate(bias) if 'msa.v.bias' in str(item)]
        # bias_v = torch.tensor([bias[3*i+2].item() for i in range(12)])

        # weights_q = torch.stack([weights[3*i] for i in range(12)])
        # weights_k = torch.stack([weights[3*i+1] for i in range(12)])
        # weights_v = torch.stack([weights[3*i+2] for i in range(12)])

        # bias_q = torch.stack([bias[3*i] for i in range(12)])
        # bias_k = torch.stack([bias[3*i+1] for i in range(12)])
        # bias_v = torch.stack([bias[3*i+2] for i in range(12)])

        # bias_q = torch.cat([bias[3*i] for i in range(12)], dim=0)
        # bias_k = torch.cat([bias[3*i+1] for i in range(12)], dim=0)
        # bias_v = torch.cat([bias[3*i+2] for i in range(12)], dim=0)
        # print("bias:", bias)
        # print("bias[0]:", bias[0])
        # print("bias[0].shape:", bias[0].shape)
        # print("len(bias):", len(bias)) 