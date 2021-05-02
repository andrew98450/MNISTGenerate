import torch

class NetG(torch.nn.Module):
    
    def __init__(self):
        super(NetG, self).__init__()
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128 * 7 * 7),
            torch.nn.BatchNorm1d(128 * 7 * 7),
            torch.nn.ReLU())
        self.deconv_layer = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(64, 1, 3, padding=1, bias=False),
            torch.nn.Tanh())
    
    def forward(self, inputs):
        outputs = self.fc_layer(inputs)
        outputs = outputs.view(outputs.shape[0], 128, 7, 7)
        outputs = self.deconv_layer(outputs)
        return outputs


class NetD(torch.nn.Module):
    
    def __init__(self):
        super(NetD, self).__init__()
        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),            
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(negative_slope=0.2))
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 7 * 7, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid())
    
    def forward(self, inputs):
        outputs = self.conv_layer(inputs)
        outputs = self.fc_layer(outputs)
        return outputs
    
