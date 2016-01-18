import math

import chainer
import chainer.functions as F
import numpy as np


class CNN_segment(chainer.FunctionSet):
    """2-layer convolutional autoencoder for 1-channel data segmentation(Hourglass network)"""

    def __init__(self, n_in=1, n_out=1, stride1=2, stride2=2):
        w = math.sqrt(2)
        super(CNN_segment, self).__init__(
            encode1=F.Convolution2D(n_in, 24, ksize=5, wscale=w, stride=stride1, dtype=np.float32),
            encode2=F.Convolution2D(24, 48, ksize=6, wscale=w, stride=stride2, dtype=np.float32),
            decode1=F.Deconvolution2D(48, 24, ksize=6, wscale=w, stride=stride2),
            decode2=F.Deconvolution2D(24, n_out, ksize=5, wscale=w, stride=stride1),
        )

    def forward(self, x_data, t_data, train=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)

        h = F.sigmoid(self.encode1(x))
        h = F.sigmoid(self.encode2(h))
        h = F.sigmoid(self.decode1(h))
        y = F.sigmoid(self.decode2(h))
        
        return F.mean_squared_error(y, t)
        
        
    def predict(self, x_data, train=False):
        x = chainer.Variable(x_data)

        h = F.sigmoid(self.encode1(x))
        h = F.sigmoid(self.encode2(h))
        h = F.sigmoid(self.decode1(h))
        y = F.sigmoid(self.decode2(h))
        
        return y.data

    
class CNN_segment3(chainer.FunctionSet):
    """3-layer convolutional autoencoder for 1-channel data segmentation(Hourglass network)"""

    def __init__(self, n_in=1, n_out=1, stride1=2, stride2=2, stride3=2):
        w = math.sqrt(2)
        super(CNN_segment3, self).__init__(
            encode1=F.Convolution2D(n_in, 24, ksize=5, wscale=w, stride=stride1, dtype=np.float32),
            encode2=F.Convolution2D(24, 48, ksize=6, wscale=w, stride=stride2, dtype=np.float32),
            encode3=F.Convolution2D(48, 96, ksize=7, wscale=w, stride=stride3, dtype=np.float32),
            decode1=F.Deconvolution2D(96, 48, ksize=7, wscale=w, stride=stride3),
            decode2=F.Deconvolution2D(48, 24, ksize=6, wscale=w, stride=stride2),
            decode3=F.Deconvolution2D(24, n_out, ksize=5, wscale=w, stride=stride1),
        )

    def forward(self, x_data, t_data, train=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)

        h = F.sigmoid(self.encode1(x))
        h = F.sigmoid(self.encode2(h))
        h = F.sigmoid(self.encode3(h))
        h = F.sigmoid(self.decode1(h))
        h = F.sigmoid(self.decode2(h))
        y = F.sigmoid(self.decode3(h))
        
        return F.mean_squared_error(y, t)
        
        
    def predict(self, x_data, train=False):
        x = chainer.Variable(x_data)

        h = F.sigmoid(self.encode1(x))
        h = F.sigmoid(self.encode2(h))
        h = F.sigmoid(self.encode3(h))
        h = F.sigmoid(self.decode1(h))
        h = F.sigmoid(self.decode2(h))
        y = F.sigmoid(self.decode3(h))
        
        return y.data

class CNN_segment4(chainer.FunctionSet):
    """4-layer convolutional autoencoder for 1-channel data segmentation(Hourglass network)"""

    def __init__(self, n_in=1, n_out=1, stride1=2, stride2=2, stride3=2, stride4=1):
        w = math.sqrt(2)
        super(CNN_segment4, self).__init__(
            encode1=F.Convolution2D(n_in, 24, ksize=5, wscale=w, stride=stride1, dtype=np.float32),
            encode2=F.Convolution2D(24, 48, ksize=6, wscale=w, stride=stride2, dtype=np.float32),
            encode3=F.Convolution2D(48, 96, ksize=5, wscale=w, stride=stride3, dtype=np.float32),
            encode4=F.Convolution2D(96, 144, ksize=6, wscale=w, stride=stride4, dtype=np.float32),
            decode1=F.Deconvolution2D(144, 96, ksize=6, wscale=w, stride=stride4),
            decode2=F.Deconvolution2D(96, 48, ksize=5, wscale=w, stride=stride3),
            decode3=F.Deconvolution2D(48, 24, ksize=6, wscale=w, stride=stride2),
            decode4=F.Deconvolution2D(24, n_out, ksize=5, wscale=w, stride=stride1),
        )

    def forward(self, x_data, t_data, train=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)

        h = F.sigmoid(self.encode1(x))
        h = F.sigmoid(self.encode2(h))
        h = F.sigmoid(self.encode3(h))
        h = F.sigmoid(self.encode4(h))
        h = F.sigmoid(self.decode1(h))
        h = F.sigmoid(self.decode2(h))
        h = F.sigmoid(self.decode3(h))
        y = F.sigmoid(self.decode4(h))
        
        return F.mean_squared_error(y, t)
        
        
    def predict(self, x_data, train=False):
        x = chainer.Variable(x_data)

        h = F.sigmoid(self.encode1(x))
        h = F.sigmoid(self.encode2(h))
        h = F.sigmoid(self.encode3(h))
        h = F.sigmoid(self.encode4(h))
        h = F.sigmoid(self.decode1(h))
        h = F.sigmoid(self.decode2(h))
        h = F.sigmoid(self.decode3(h))
        y = F.sigmoid(self.decode4(h))
        
        return y.data
