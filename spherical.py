import torch
import torch.nn as nn
import torch.nn.functional as F
import healpy as hp
import numpy as np

class sphericalConv(nn.Module):
    def __init__(self, NSIDE, in_channels, out_channels, bias=True, nest=True):
        """Convolutional layer as defined in Krachmalnicoff & Tomasi (A&A, 2019, 628, A129)

        Parameters
        ----------
        NSIDE : int
            HEALPix NSIDE
        in_channels : int
            Number of channels of the input. The size is [B,C_in,N], with B batches, 
            C_in channels and N pixels in the HEALPix pixelization
        out_channels : int
            Number of channels of the output. The size is [B,C_out,N], with B batches, 
            C_out channels and N pixels in the HEALPix pixelization
        bias : bool, optional
            Add bias, by default True
        nest : bool, optional
            Used nested mapping, by default True
            Always use nested mapping if pooling layers are used.
        """
        super(sphericalConv, self).__init__()

        self.NSIDE = NSIDE
        self.npix = hp.nside2npix(self.NSIDE)
        self.nest = nest

        self.neighbours = torch.zeros(9 * self.npix, dtype=torch.long)
        self.weight = torch.ones(9 * self.npix, dtype=torch.float32)

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=9, bias=bias)

        for i in range(self.npix):
            neighbours = hp.pixelfunc.get_all_neighbours(self.NSIDE, i, nest=nest)
            neighbours = np.insert(neighbours, 4, i)

            ind = np.where(neighbours == -1)[0]
            neighbours[ind] = self.npix            

            self.neighbours[9*i:9*i+9] = torch.tensor(neighbours)

        self.zeros = torch.zeros((1, 1, 1))

        nn.init.kaiming_normal_(self.conv.weight)        
        if (bias):
            nn.init.constant_(self.conv.bias, 0.0)
        
    def forward(self, x):

        x2 = F.pad(x, (0,1,0,0,0,0), mode='constant', value=0.0)
                
        vec = x2[:, :, self.neighbours]
        
        tmp = self.conv(vec)

        return tmp

class sphericalDown(nn.Module):    
    def __init__(self, NSIDE):
        """Average pooling layer

        Parameters
        ----------
        NSIDE : int
            HEALPix NSIDE
        """
        super(sphericalDown, self).__init__()
        
        self.pool = nn.AvgPool1d(4)
                
    def forward(self, x):
                
        return self.pool(x)

class sphericalUp(nn.Module):
    def __init__(self, NSIDE):
        """Upsampling pooling layer

        Parameters
        ----------
        NSIDE : int
            HEALPix NSIDE
        """
        super(sphericalUp, self).__init__()
                
    def forward(self, x):
        
        return torch.repeat_interleave(x, 4, dim=-1)