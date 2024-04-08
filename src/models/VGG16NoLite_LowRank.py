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
    w = [wd['classifier.0.weight'], wd['classifier.4.weight'], wd['classifier.8.weight']]
    b = [wd['classifier.0.bias'], wd['classifier.4.bias'], wd['classifier.8.bias']]
    base_dict = {
        'classifier.0.weight' : wd['classifier.0.weight'],
        'classifier.0.bias' : wd['classifier.0.bias'],
        'classifier.4.weight' : wd['classifier.4.weight'],
        'classifier.4.bias' : wd['classifier.4.bias'],
        'classifier.8.weight' : wd['classifier.8.weight'],
        'classifier.8.bias' : wd['classifier.8.bias']
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

class VGG16NoLite_LowRank(nn.Module):
    def __init__(self, weights, bias, num = 10, rank = -1):
        super(VGG16NoLite_LowRank, self).__init__()
        self.feature = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d( kernel_size=2, stride=2),

            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), 
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d( kernel_size=2, stride=2),

            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d( kernel_size=2, stride=2),

            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),   
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),   
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2))

            ## Layer 1 - 64 channels
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # BatchNorm layer after Convolution
            nn.ReLU(inplace=True),
            
            ## Layer 2 - 64 channels
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # BatchNorm layer
            nn.ReLU(inplace=True),
            nn.MaxPool2d( kernel_size=2, stride=2),

            ## Layer 3 - 128 channels
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # BatchNorm layer
            nn.ReLU(inplace=True),

            ## Layer 4 - 128 channels
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128),  # BatchNorm layer
            nn.ReLU(inplace=True),
            nn.MaxPool2d( kernel_size=2, stride=2),

            ## Layer 5 - 256 channels
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # BatchNorm layer
            nn.ReLU(inplace=True),

            ## Layer 6 - 256 channels
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # BatchNorm layer
            nn.ReLU(inplace=True),

            ## Layer 7 - 256 channels
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # BatchNorm layer
            nn.ReLU(inplace=True),
            nn.MaxPool2d( kernel_size=2, stride=2),

            ## Layer 8 - 512 channels
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # BatchNorm layer
            nn.ReLU(inplace=True),   

            ## Layer 9 - 512 channels
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # BatchNorm layer
            nn.ReLU(inplace=True),

            ## Layer 10 - 512 channels
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # BatchNorm layer
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ## Layer 11 - 512 channels
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # BatchNorm layer
            nn.ReLU(inplace=True),   

            ## Layer 12 - 512 channels
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # BatchNorm layer
            nn.ReLU(inplace=True),

            ## Layer 13 - 512 channels
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # BatchNorm layer
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )

        # self.classifier = nn.Sequential(LowRankLinear(512, 4096, weights[0], bias[0], rank = rank),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Dropout(0.5),
        #                                 LowRankLinear(4096, 4096, weights[1], bias[1], rank = rank),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Dropout(0.5),
        #                                 LowRankLinear(4096, 10, weights[2], bias[2], rank = rank))
                                        
        self.classifier = nn.Sequential(LowRankLinear(25088, 4096, weights[0], bias[0], rank = rank),
                                        nn.BatchNorm1d(4096),  # BatchNorm layer
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5),
                                        LowRankLinear(4096, 4096, weights[1], bias[1], rank = rank),
                                        nn.BatchNorm1d(4096),  # BatchNorm layer
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5),
                                        LowRankLinear(4096, 10, weights[2], bias[2], rank = rank))
                                        # Softmax is not needed as we are using Cross-Entropy loss
                          
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x