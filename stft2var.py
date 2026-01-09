import os
import numpy as np
import random

# Directories containing the STFT data
stft_output_dir = 'stft_data'
mag_mixed_dir = os.path.join(stft_output_dir, 'magnitude', 'mixed_audio')
phase_mixed_dir = os.path.join(stft_output_dir, 'phase', 'mixed_audio')
mag_clean_speech_dir = os.path.join(stft_output_dir, 'magnitude', 'clean_speech')
phase_clean_speech_dir = os.path.join(stft_output_dir, 'phase', 'clean_speech')
mag_clean_noise_dir = os.path.join(stft_output_dir, 'magnitude', 'clean_noise')
phase_clean_noise_dir = os.path.join(stft_output_dir, 'phase', 'clean_noise')

# Get a list of all mixed audio STFT magnitude files (as a representative list for all mixes)
mixed_mag_files = sorted([f for f in os.listdir(mag_mixed_dir) if f.endswith('_mag.npy')])
random.shuffle(mixed_mag_files) # Shuffle to ensure random distribution for splits

# Define split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

num_files = len(mixed_mag_files)
num_train = int(num_files * train_ratio)
num_val = int(num_files * val_ratio)
num_test = num_files - num_train - num_val # Ensure all files are allocated

# Split the filenames
train_files = mixed_mag_files[:num_train]
val_files = mixed_mag_files[num_train : num_train + num_val]
test_files = mixed_mag_files[num_train + num_val :]

print(f"Total STFT files: {num_files}")
print(f"Training set size: {len(train_files)}")
print(f"Validation set size: {len(val_files)}")
print(f"Test set size: {len(test_files)}")

# Store file paths for easy access (example for one set)
def get_filepaths(file_list, mag_dir, phase_dir, clean_speech_mag_dir, clean_speech_phase_dir, clean_noise_mag_dir, clean_noise_phase_dir):
    data = []
    for filename in file_list:
        mix_id = filename.replace('_mag.npy', '') # e.g., 'mix_001'

        # Extract base mix_id (e.g., 'mix_001' from 'mix_001_mic1') if files have mic suffixes
        # Mixed audio files are named 'mix_XXX_mag.npy', so the mix_id is just 'mix_XXX'
        # For clean files, it will be 'mix_XXX_speech_mic1_mag.npy', we need 'mix_XXX'
        if '_speech_mic1' in mix_id or '_speech_mic2' in mix_id or '_noise_mic1' in mix_id or '_noise_mic2' in mix_id:
            mix_id = '_'.join(mix_id.split('_')[:2]) # Get 'mix_XXX'

        data.append({
            'mixed_mag_path': os.path.join(mag_mixed_dir, f"{mix_id}_mag.npy"), # Reconstruct exact filename
            'mixed_phase_path': os.path.join(phase_mixed_dir, f"{mix_id}_phase.npy"),
            'clean_speech_mic1_mag_path': os.path.join(clean_speech_mag_dir, f"{mix_id}_speech_mic1_mag.npy"),
            'clean_speech_mic1_phase_path': os.path.join(clean_speech_phase_dir, f"{mix_id}_speech_mic1_phase.npy"),
            'clean_speech_mic2_mag_path': os.path.join(clean_speech_mag_dir, f"{mix_id}_speech_mic2_mag.npy"),
            'clean_speech_mic2_phase_path': os.path.join(clean_speech_phase_dir, f"{mix_id}_speech_mic2_phase.npy"),
            'clean_noise_mic1_mag_path': os.path.join(clean_noise_mag_dir, f"{mix_id}_noise_mic1_mag.npy"),
            'clean_noise_mic1_phase_path': os.path.join(clean_noise_phase_dir, f"{mix_id}_noise_mic1_phase.npy"),
            'clean_noise_mic2_mag_path': os.path.join(clean_noise_mag_dir, f"{mix_id}_noise_mic2_mag.npy"),
            'clean_noise_mic2_phase_path': os.path.join(clean_noise_phase_dir, f"{mix_id}_noise_mic2_phase.npy"),
        })
    return data

train_data = get_filepaths(train_files, mag_mixed_dir, phase_mixed_dir, mag_clean_speech_dir, phase_clean_speech_dir, mag_clean_noise_dir, phase_clean_noise_dir)
val_data = get_filepaths(val_files, mag_mixed_dir, phase_mixed_dir, mag_clean_speech_dir, phase_clean_speech_dir, mag_clean_noise_dir, phase_clean_noise_dir)
test_data = get_filepaths(test_files, mag_mixed_dir, phase_mixed_dir, mag_clean_speech_dir, phase_clean_speech_dir, mag_clean_noise_dir, phase_clean_noise_dir)

print("Data split into training, validation, and test sets. Each entry in the sets contains paths to magnitude and phase STFT data for mixed, clean speech, and clean noise components.")