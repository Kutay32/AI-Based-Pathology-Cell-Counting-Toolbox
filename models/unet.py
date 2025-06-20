"""
Attention U-Net model for cell segmentation in pathology images.
"""

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate, Activation, BatchNormalization
from tensorflow.keras.layers import Multiply, Add

def attention_block(x, g, inter_channels):
    """
    Attention gate that helps the model focus on relevant regions.
    
    Args:
        x: Input feature map from the skip connection
        g: Gating signal from the previous layer
        inter_channels: Number of intermediate channels
        
    Returns:
        Attention-weighted feature map
    """
    theta_x = Conv2D(inter_channels, (1, 1))(x)
    phi_g = Conv2D(inter_channels, (1, 1))(g)
    add = Add()([theta_x, phi_g])
    relu = Activation('relu')(add)
    psi = Conv2D(1, (1, 1))(relu)
    sigmoid = Activation('sigmoid')(psi)
    return Multiply()([x, sigmoid])

def build_attention_unet(input_shape, num_classes, filters=64, dropout=0.1):
    """
    Build an Attention U-Net model for cell segmentation.
    
    Args:
        input_shape: Shape of the input images
        num_classes: Number of output classes
        filters: Number of filters in the first layer
        dropout: Dropout rate
        
    Returns:
        Compiled Attention U-Net model
    """
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Dropout(dropout)(c1)
    c1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(filters*2, (3, 3), activation='relu', padding='same')(p1)
    c2 = Dropout(dropout)(c2)
    c2 = Conv2D(filters*2, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(filters*4, (3, 3), activation='relu', padding='same')(p2)
    c3 = Dropout(dropout)(c3)
    c3 = Conv2D(filters*4, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(filters*8, (3, 3), activation='relu', padding='same')(p3)
    c4 = Dropout(dropout)(c4)
    c4 = Conv2D(filters*8, (3, 3), activation='relu', padding='same')(c4)

    # Decoder
    u3 = Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding='same')(c4)
    a3 = attention_block(c3, u3, filters*4)
    u3 = concatenate([u3, a3])
    c5 = Conv2D(filters*4, (3, 3), activation='relu', padding='same')(u3)
    c5 = Dropout(dropout)(c5)
    c5 = Conv2D(filters*4, (3, 3), activation='relu', padding='same')(c5)

    u2 = Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding='same')(c5)
    a2 = attention_block(c2, u2, filters*2)
    u2 = concatenate([u2, a2])
    c6 = Conv2D(filters*2, (3, 3), activation='relu', padding='same')(u2)
    c6 = Dropout(dropout)(c6)
    c6 = Conv2D(filters*2, (3, 3), activation='relu', padding='same')(c6)

    u1 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c6)
    a1 = attention_block(c1, u1, filters)
    u1 = concatenate([u1, a1])
    c7 = Conv2D(filters, (3, 3), activation='relu', padding='same')(u1)
    c7 = Dropout(dropout)(c7)
    c7 = Conv2D(filters, (3, 3), activation='relu', padding='same')(c7)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c7)

    model = Model(inputs, outputs)
    return model