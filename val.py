from generator import *

val_generator = SpeechSeparationDataGenerator(
    data_paths=val_data,
    n_bins=n_bins,
    target_n_bins=target_n_bins,
    n_frames_example=n_frames_example,
    batch_size=batch_size,
    shuffle=False
)

print("Validation Data Generator instantiated.")