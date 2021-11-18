import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Layer, concatenate, Input
from tensorflow.keras.models import Model


class ResNetBlock (Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)
        self.conv2d_relu = layers.Conv2D(filters, kernel_size, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-5))
        self.conv2d = layers.Conv2D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(1e-5))
        self.batchnorm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.add = layers.Add()
        
    def call(self, input):
        x = self.conv2d_relu(input)
        x = self.batchnorm(x)
        x = self.conv2d(x)
        out = self.add([input, x])
        out = self.relu(out)
        out = self.batchnorm(out)
        return out 

class UpDownBlock(Layer):
    def __init__(self, filters, kernel_size, nrblock, **kwargs):
        super(UpDownBlock, self).__init__(**kwargs) 
        self.conv2d = layers.Conv2D(filters, kernel_size, kernel_regularizer=regularizers.l2(1e-5), padding='same') 
        self.max_pooling = layers.MaxPooling2D((2,2))
        self.nrblock = nrblock
        self.resnetblock = ResNetBlock(filters, kernel_size)

    def call(self, input):
        x = self.conv2d(input)
        for i in range(self.nrblock):
            x = self.resnetblock(x)
        out = self.max_pooling(x)
        return out, x

class DownUpBlock(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(DownUpBlock, self).__init__(**kwargs)
        self.resnetblock = ResNetBlock(filters, kernel_size)
        self.conv2d = layers.Conv2D(filters, kernel_size, activation='relu', kernel_regularizer=regularizers.l2(1e-5), padding='same')
        self.convtr2d = layers.Conv2DTranspose(filters, (2,2), strides=(2,2), padding='same')

    def call (self, left_input, right_input):
        x = concatenate([self.convtr2d(left_input), right_input], axis=3) #up
        x = self.conv2d(x)
        out = self.resnetblock(x)
        return out

def ResUnetModel(input_size):

    config = {'filters': [32, 64, 128, 256],
            'kernel_size': [(1,1), (3,3)],
            'nrblocks': [3,4,6,3]}

    inputs = Input(input_size)
    x = inputs
    d={}

    # 4 forward blocks
    for i in range(len(config['filters'])):
        x, x1 = UpDownBlock(config['filters'][i],
                            config['kernel_size'][1],
                            config['nrblocks'][i])(x)
        d[i]=x1

    #bottom blocks
    x = layers.Conv2D(512, config['kernel_size'][1], activation='relu', kernel_regularizer=regularizers.l2(1e-5), padding='same')(x)
    x = ResNetBlock(512, config['kernel_size'][1])(x)       

    #reverse blocks
    for i in reversed(range(len(config['filters']))):
        x = DownUpBlock(config['filters'][i],
                        config['kernel_size'][1])(x,d[i])
        x = layers.Conv2D(config['filters'][i], config['kernel_size'][1], activation='relu', kernel_regularizer=regularizers.l2(1e-5), padding='same')(x)
    
    outputs = layers.Conv2D(1,config['kernel_size'][0],activation='sigmoid', kernel_regularizer=regularizers.l2(1e-5), padding='same') (x)

    return Model(inputs=[inputs], outputs=[outputs])

    