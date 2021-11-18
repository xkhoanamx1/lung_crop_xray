import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

# Create residual block based on ResNet definition
def res_block(x, kernel_size, filters):
    fx = Conv2D(filters, kernel_size, activation='relu', kernel_regularizer=regularizers.l2(1e-5), padding='same')(x)
    fx = BatchNormalization()(fx)
    fx = Conv2D(filters, kernel_size, kernel_regularizer=regularizers.l2(1e-5), padding='same')(fx)
    out = Add()([x,fx])
    out = ReLU()(out)
    out = BatchNormalization()(out)
    return out

class layers_model(): 
    def conv_3x3(x, filters):
        return Conv2D(filters, (3,3), activation='relu', kernel_regularizer=regularizers.l2(1e-5), padding='same')(x)
    def conv_1x1(x, filters):
        return Conv2D(filters, (1,1), activation='sigmoid', kernel_regularizer=regularizers.l2(1e-5), padding='same')(x)
    def max_pooling(x):
        return MaxPooling2D((2,2))(x)
    def convTr_2x2(x, filters):
        return Conv2DTranspose(filters, (2,2), strides=(2,2), padding='same')(x)
    def dropout(x):
        return Dropout(0.1)(x)
 
def resnet34_unet_model(input_size): #e.g input_size = (256,256,1) for grayscale
    inputs = Input(input_size)
        
    y1 = layers_model.conv_3x3(inputs,32)
    y1 = res_block(y1, (3,3), 32)
    y1 = res_block(y1, (3,3), 32)
    y1 = res_block(y1, (3,3), 32)
    pool1 = layers_model.max_pooling(y1)
    
    y2 = layers_model.conv_3x3(pool1,64)
    y2 = res_block(y2, (3,3), 64)
    y2 = res_block(y2, (3,3), 64)
    y2 = res_block(y2, (3,3), 64)
    y2 = res_block(y2, (3,3), 64)
    pool2 = layers_model.max_pooling(y2)
    
    y3 = layers_model.conv_3x3(pool2,128)
    y3 = res_block(y3, (3,3), 128)
    y3 = res_block(y3, (3,3), 128)
    y3 = res_block(y3, (3,3), 128)
    y3 = res_block(y3, (3,3), 128)
    y3 = res_block(y3, (3,3), 128)
    y3 = res_block(y3, (3,3), 128)
    pool3 = layers_model.max_pooling(y3)
    
    y4 = layers_model.conv_3x3(pool3,256)
    y4 = res_block(y4, (3,3), 256)
    y4 = res_block(y4, (3,3), 256)
    y4 = res_block(y4, (3,3), 256)
    pool4 = layers_model.max_pooling(y4)

    y5 = layers_model.conv_3x3(pool4,512)
    y5 = res_block(y5, (3,3), 512)
#     y5 = layers_model.dropout(y5)

    up6 = concatenate([layers_model.convTr_2x2(y5,256), y4], axis=3)
    y6 = layers_model.conv_3x3(up6,256)
    y6 = res_block(y6, (3,3), 256)
    y6 = layers_model.conv_3x3(y6,256)
#     y6 = layers_model.dropout(y6)

    up7 = concatenate([layers_model.convTr_2x2(y6,128), y3], axis=3)
    y7 = layers_model.conv_3x3(up7,128)
    y7 = res_block(y7, (3,3), 128)
    y7 = layers_model.conv_3x3(y7,128)
#     y7 = layers_model.dropout(y7)

    up8 = concatenate([layers_model.convTr_2x2(y7,64), y2], axis=3)
    y8 = layers_model.conv_3x3(up8,64)
    y8 = res_block(y8, (3,3), 64)
    y8 = layers_model.conv_3x3(y8,64)
#     y8 = layers_model.dropout(y8)

    up9 = concatenate([layers_model.convTr_2x2(y8,32), y1], axis=3)
    y9 = layers_model.conv_3x3(up9,32)
    y9 = res_block(y9, (3,3), 32)
    y9 = layers_model.conv_3x3(y9,32)
        
    outputs = layers_model.conv_1x1(y9, 1)
        
    return Model(inputs=[inputs], outputs=[outputs])
 
# model = resnet34_unet_model((256,256,1))
# model.summary()
