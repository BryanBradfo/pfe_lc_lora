import torch.nn as nn
import torch, os
from src.compression.LowRankLinear import LowRankLinear

def getBase(model, basepath=""):
    """
    @param model : The original VGG16.
    
    @return The weights and bias needed to act as 
        the base for the low-rank version of the custom linear layers.
    """

    wd = model.state_dict()
    w = [wd['classifier.0.weight'], wd['classifier.3.weight'], wd['classifier.6.weight']]
    b = [wd['classifier.0.bias'], wd['classifier.3.bias'], wd['classifier.6.bias']]
    base_dict = {
        'classifier.0.weight' : wd['classifier.0.weight'],
        'classifier.0.bias' : wd['classifier.0.bias'],
        'classifier.3.weight' : wd['classifier.3.weight'],
        'classifier.3.bias' : wd['classifier.3.bias'],
        'classifier.6.weight' : wd['classifier.6.weight'],
        'classifier.6.bias' : wd['classifier.6.bias']
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

class VGG16_LowRank(nn.Module):
    def __init__(self, weights, bias, num = 10, rank = -1):
        super(VGG16_LowRank, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d( kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d( kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d( kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),   
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(LowRankLinear(512, 4096, weights[0], bias[0], rank = rank),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5),
                                        LowRankLinear(4096, 4096, weights[1], bias[1], rank = rank),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5),
                                        LowRankLinear(4096, 10, weights[2], bias[2], rank = rank))
                                        

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x