import torch.nn as nn
import os, torch
from src.compression.LowRankLinear import LowRankLinear


# TODO: getBase should not be model specific, should be included across the board.

def getBase(model, basepath=""):
    """
    @param model : The original AlexNet.
    
    @return The weights and bias needed to act as 
        the base for the low-rank version of the custom linear layers.
    """

    wd = model.state_dict()
    w = [wd['classifier.1.weight'], wd['classifier.4.weight']]
    b = [wd['classifier.1.bias'], wd['classifier.4.bias']]

    # Save base weights.
    base_dict = {
        'classifier.1.weight' : wd['classifier.1.weight'],
        'classifier.1.bias' : wd['classifier.1.bias'],
        'classifier.4.weight' : wd['classifier.4.weight'],
        'classifier.4.bias' : wd['classifier.4.bias']
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

class AlexNet_LowRank(nn.Module):   
    def __init__(self, weights : list, bias : list, num=10, rank = -1):
        """
        @param weights : List of initial bases for the loRA linear layers, kept as a parameter.
        @param bias : List of initial biases for the loRA linear layers, kept as a parameter.
        @param rank : The rank of the original model to be kept.
        """
        super(AlexNet_LowRank, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d( kernel_size=2, stride=2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),                         
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),                         
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d( kernel_size=2, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            LowRankLinear(32*12*12, 2048, weights[0], bias[0], rank = rank),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            LowRankLinear(2048, 1024, weights[1], bias[1], rank = rank),
            nn.ReLU(inplace=True),
            nn.Linear(1024,num),
        )
    
    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1,32*12*12)
        x = self.classifier(x)
        return x