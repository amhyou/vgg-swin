"""
U-Net Architecture Implementation for Biomedical Image Segmentation

This module implements the U-Net architecture as described in the paper:
"U-Net: Convolutional Networks for Biomedical Image Segmentation"
by Olaf Ronneberger, Philipp Fischer, and Thomas Brox (2015)

The U-Net is a convolutional neural network designed specifically for biomedical
image segmentation. It consists of a contracting path (encoder) that captures
context and a symmetric expanding path (decoder) that enables precise localization.
The key innovation is the use of skip connections between corresponding layers
in the encoder and decoder, which helps preserve spatial information lost during
downsampling.

Original paper: https://arxiv.org/abs/1505.04597
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate



# --- U-NET ARCHITECTURE  ---
def get_unet(input_size=(224, 224, 1)):
    """
    Create a U-Net model for image segmentation.

    The U-Net architecture follows the design from the original paper:
    - Encoder: 4 downsampling blocks with increasing filter counts (32, 64, 128, 256)
    - Bridge: Bottleneck layer with 512 filters
    - Decoder: 4 upsampling blocks with decreasing filter counts (256, 128, 64, 32)
    - Skip connections: Concatenation of encoder features with decoder features
    - Output: Single-channel sigmoid activation for binary segmentation

    Args:
        input_size (tuple): Input image dimensions (height, width, channels).
                           Default is (224, 224, 1) for grayscale images.

    Returns:
        tensorflow.keras.Model: Compiled U-Net model ready for training/inference.
    """
    inputs = Input(input_size)

    # --- Encoder (Downsampling) ---
    # The encoder captures contextual information through successive downsampling
    # Each block consists of two 3x3 convolutions followed by max pooling

    # Block 1: 32 filters
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Block 2: 64 filters
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Block 3: 128 filters
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Block 4: 256 filters
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # --- Bridge (Bottom) ---
    # The bridge is the lowest resolution layer, capturing the most abstract features
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    # --- Decoder (Upsampling with Transpose) ---
    # The decoder reconstructs the segmentation map using transpose convolutions
    # Skip connections from encoder layers provide spatial information

    # Up Block 1: 256 filters (upsample from 512 to match encoder Block 4)
    # Note: The original U-Net paper uses upsampling + convolution, but this implementation
    # uses Conv2DTranspose for efficiency. The skip connection concatenates features.
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    # Up Block 2: 128 filters (upsample to match encoder Block 3)
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    # Up Block 3: 64 filters (upsample to match encoder Block 2)
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    # Up Block 4: 32 filters (upsample to match encoder Block 1)
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    # --- Output ---
    # Final 1x1 convolution with sigmoid activation for binary segmentation
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])