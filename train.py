import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Activation, Cropping2D
import numpy as np

# Model parameters
n_fft = 512       # Window size
hop_length = 128  # Hop length (stride)
sr = 16000

# Determine the number of frequency bins (n_bins) which is n_fft // 2 + 1
#n_bins = n_fft // 2 + 1 # This will be 257 for n_fft=512 ##added
N_BINS = 256


# Use an example audio length (e.g., 5 seconds) to determine a representative N_FRAMES.
# Ensure N_FRAMES_EXAMPLE is a multiple of 4 to simplify initial dimension matching.
# For 5 seconds at 16kHz: 16000 * 5 = 80000 samples.
# n_frames = 1 + (80000 - n_fft) // hop_length = 1 + (80000 - 512) // 128 = 1 + 621 = 622 frames.
# To make it a multiple of 4 for pooling operations: use 620 or 624.
# Let's use 620 frames for consistency in this example to ensure pooling and upsampling align.
N_FRAMES_EXAMPLE = 620 # (620 // 2 = 310, 310 // 2 = 155, 155 * 2 = 310)

# Input shape for a single microphone's magnitude spectrogram
# (frequency_bins, time_frames, channels)
#input_shape = (n_bins, N_FRAMES_EXAMPLE, 1) ##added
input_shape = (256, N_FRAMES_EXAMPLE, 2)

def build_separation_model(input_shape, num_filters=16):
    inputs = Input(shape=input_shape) # Expected input: (n_bins=257, N_FRAMES_EXAMPLE=620, 1) for one mic's magnitude

    # Encoder path
    # Block 1
    conv1_block = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(inputs)
    conv1_block = BatchNormalization()(conv1_block)
    conv1_skip = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(conv1_block)
    conv1_skip = BatchNormalization()(conv1_skip) # Output shape: (None, 257, 620, num_filters)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_skip) # Output shape: (None, 128, 310, num_filters)

    # Block 2
    conv2_block = Conv2D(num_filters*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2_block = BatchNormalization()(conv2_block)
    conv2_skip = Conv2D(num_filters*2, (3, 3), activation='relu', padding='same')(conv2_block)
    conv2_skip = BatchNormalization()(conv2_skip) # Output shape: (None, 128, 310, num_filters*2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_skip) # Output shape: (None, 64, 155, num_filters*2)

    # Block 3 (Bottleneck)
    conv3_block = Conv2D(num_filters*4, (3, 3), activation='relu', padding='same')(pool2)
    conv3_block = BatchNormalization()(conv3_block)
    conv3 = Conv2D(num_filters*4, (3, 3), activation='relu', padding='same')(conv3_block)
    conv3 = BatchNormalization()(conv3) # Output shape: (None, 64, 155, num_filters*4)

    # Decoder path
    # Up-Block 1
    up4_upsample = UpSampling2D(size=(2, 2))(conv3) # Output shape: (None, 128, 310, num_filters*4)
    up4_conv = Conv2D(num_filters*2, (2, 2), activation='relu', padding='same')(up4_upsample) # Output shape: (None, 128, 310, num_filters*2)

    # No cropping needed for conv2_skip as N_FRAMES_EXAMPLE=620 ensures width matches (310 vs 310)
    # and n_bins=257 -> 128 for pool1/conv2_skip, and (257//4)*2 = 64*2 = 128 for upsampled, so height matches.
    merged4 = concatenate([conv2_skip, up4_conv], axis=-1)
    conv4_block = Conv2D(num_filters*2, (3, 3), activation='relu', padding='same')(merged4)
    conv4_block = BatchNormalization()(conv4_block)
    conv4 = Conv2D(num_filters*2, (3, 3), activation='relu', padding='same')(conv4_block)
    conv4 = BatchNormalization()(conv4) # Output shape: (None, 128, 310, num_filters*2)

    # Up-Block 2
    up5_upsample = UpSampling2D(size=(2, 2))(conv4) # Output shape: (None, 256, 620, num_filters*2)
    up5_conv = Conv2D(num_filters, (2, 2), activation='relu', padding='same')(up5_upsample) # Output shape: (None, 256, 620, num_filters)

    # Cropping for conv1_skip to match up5_conv
    # conv1_skip shape: (None, 257, 620, num_filters)
    # up5_conv shape: (None, 256, 620, num_filters)
    # Height mismatch: 257 vs 256. Need to crop conv1_skip height by 1.
    # Width matches: 620 vs 620.
    # Cropping: ((top_crop, bottom_crop), (left_crop, right_crop))
    # cropped_conv1 = Cropping2D(cropping=((1, 0), (0, 0)))(conv1_skip) # Crop 1 from bottom of height ##added

    # merged5 = concatenate([cropped_conv1, up5_conv], axis=-1) ##added
    merged5 = concatenate([conv1_skip, up5_conv], axis=-1)

    conv5_block = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(merged5)
    conv5_block = BatchNormalization()(conv5_block)
    conv5 = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(conv5_block)
    conv5 = BatchNormalization()(conv5) # Output shape: (None, 256, 620, num_filters)

    # Output layer: predict magnitude. Use 'relu' for non-negative magnitudes.
    #outputs = Conv2D(1, (1, 1), activation='relu', padding='same')(conv5) # Output shape: (None, 256, 620, 1) ##added
    outputs = Conv2D(2, (1, 1), activation='relu', padding='same')(conv5)


    # Need to ensure the output matches the target clean speech magnitude size
    # Target size is (n_bins, N_FRAMES_EXAMPLE) = (257, 620)
    # Current output is (256, 620)
    # To match target (257, 620), we need to pad the output by 1 in height or output 257 from conv5
    # The U-Net structure with MaxPooling often results in output sizes that are multiples of 2^N, where N is num pooling operations.
    # Since 257 is not divisible by 2, and we have 2 pooling layers, the output height of 256 is expected.
    # A common way to handle this is to pad the input before the U-Net, or crop the target output if permissible,
    # or use padding='valid' and calculate exact output sizes.
    # For now, let's keep the model output shape as (256, 620, 1) and acknowledge this minor mismatch with original n_bins (257).
    # The alternative is to ensure n_bins is also a multiple of 4, or pad the input spectrograms before feeding to the model.

    model = Model(inputs=inputs, outputs=outputs)

    return model

# Instantiate the model with the derived input shape for a single microphone
model = build_separation_model(input_shape=input_shape)

# Compile the model
# Using Adam optimizer, MAE loss (common for magnitude spectrogram prediction)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae', metrics=['mse'])

print("Deep learning model (U-Net-like) defined and compiled. Summary:")
model.summary()