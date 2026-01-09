import numpy as np
from stft2var import *
from customclass import *
from train import *
# Define parameters consistent with the model and preprocessing
n_bins = 257  # Original frequency bins after STFT (n_fft // 2 + 1)
target_n_bins = 256 # Model's output frequency bins after cropping
n_frames_example = N_FRAMES_EXAMPLE # Use the N_FRAMES_EXAMPLE defined in the model definition step
batch_size = 4  # Example batch size

# Instantiate the data generator for the training set
train_generator = SpeechSeparationDataGenerator(
    data_paths=train_data, # Using the train_data prepared in the data splitting step
    n_bins=n_bins,
    target_n_bins=target_n_bins,
    n_frames_example=n_frames_example,
    batch_size=batch_size,
    shuffle=True
)

print("Train Data Generator instantiated.")

# Test the generator by getting one batch
X_batch, y_batch = train_generator[0]

print(f"Shape of X_batch (input mixed magnitudes): {X_batch.shape}")
print(f"Shape of y_batch (target clean speech magnitudes): {y_batch.shape}")

# Verify data types and values (optional, for deeper inspection)
print(f"Data type of X_batch: {X_batch.dtype}")
print(f"Min/Max of X_batch: {np.min(X_batch):.4f}/{np.max(X_batch):.4f}")
print(f"Data type of y_batch: {y_batch.dtype}")
print(f"Min/Max of y_batch: {np.min(y_batch):.4f}/{np.max(y_batch):.4f}")