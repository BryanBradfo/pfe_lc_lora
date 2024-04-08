import torch.nn as nn

class VGG16NoLite(nn.Module):
    def __init__(self):
        super(VGG16NoLite, self).__init__()
        self.feature = nn.Sequential(

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

        self.classifier = nn.Sequential(nn.Linear(25088, 4096), # 512 * 7 * 7 = 25088 and 7 because of 32x32 image and 5 maxpool layers of 2x2 kernel size 
                                        nn.BatchNorm1d(4096),  # BatchNorm layer
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5),
                                        nn.Linear(4096, 4096),
                                        nn.BatchNorm1d(4096),  # BatchNorm layer
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5),
                                        nn.Linear(4096, 10))
                                        # Softmax is not needed as we are using Cross-Entropy loss
                                        

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x