import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)
    
    
class UnFlatten(nn.Module):
    def forward(self, inputs):
        return inputs.view(inputs.size(0), 14, 14, 32)
    
    
class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=128, z_dim=2):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=2)
	    nn.ReLU(),
	    nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=2)
	    nn.ReLU(),
	    nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
	    Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
	    nn.Linear(h_dim, 14 * 14 * 32)
	    nn.ReLU(),
            UnFlatten(),
	    nn.Dropout(0.25),
	    nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
 	    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2)
	    nn.ReLU(),
	    nn.Conv2d(32, image_channels, kernel_size=3, stride=1, padding=2)
	    nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return self.decoder(z), mu, logvar