import torch.nn as nn
import os
import torch
from src.compression.LowRankLinear import LowRankLinear

def getBase(model, basepath=""):
    """
    @param model : The original LeNet.
    
    @return The weights and bias needed to act as the base for the 
        low-rank version of the custom linear layers.
    """

    wd = model.state_dict()
    w = [wd['classifier.1.weight'], wd['classifier.3.weight']]
    b = [wd['classifier.1.bias'], wd['classifier.3.bias']]

    base_dict = {
        'classifier.1.weight' : wd['classifier.1.weight'],
        'classifier.1.bias' : wd['classifier.1.bias'],
        'classifier.3.weight' : wd['classifier.3.weight'],
        'classifier.3.bias' : wd['classifier.3.bias']
    }
    if basepath != "":
        if not os.path.exists(basepath):
            os.makedirs(basepath)
        fp = os.path.join(basepath, "lora_bases.pt")
        torch.save(base_dict, fp)
        
    return w, b

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

class LeNet_LowRank(nn.Module):
    def __init__(self, weights : list, bias : list, num=10, rank = -1):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),   # 28*28->32*32-->28*28
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 10*10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 5*5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LowRankLinear(16*5*5, 120, weights[0], bias[0], rank = rank),
            nn.Tanh(),
            LowRankLinear(120, 84, weights[1], bias[1], rank = rank),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        return self.classifier(self.feature(x))