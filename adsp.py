## import audio-spectograph file
## Import custom room mix

import custom_room_mix
from inputsounds import *
import soundfile as sf
import numpy as np
import os
import random
import importlib # Ensure custom_room_mix is reloaded if needed

# Reload the module one last time to be sure we have the latest version
importlib.reload(custom_room_mix)

# 1. Create a root directory named 'generated_dataset' and subdirectories
root_output_dir = 'generated_dataset'
mixed_audio_dir = os.path.join(root_output_dir, 'mixed_audio')
clean_speech_dir = os.path.join(root_output_dir, 'clean_speech')
clean_noise_dir = os.path.join(root_output_dir, 'clean_noise')

os.makedirs(mixed_audio_dir, exist_ok=True)
os.makedirs(clean_speech_dir, exist_ok=True)
os.makedirs(clean_noise_dir, exist_ok=True)

print(f"Created output directories: {mixed_audio_dir}, {clean_speech_dir}, {clean_noise_dir}")

# List of available speech and noise audio files (from previous steps)
speech_sources = {
    'male_speech': male_speech_audio,
    'female_speech': female_speech_audio
}
noise_sources = {
    'pink_noise': pink_noise_audio,
    'tones': tones_audio
}

sr = 16000 # Sampling rate

# 2. Loop 5000 times to generate 5000 unique mixed audio files
num_generations = 5000

for i in range(num_generations):
    print(f"\nGenerating mix {i+1}/{num_generations}...")

    # 3. Randomly choose one speech source and one noise source
    speech_name, selected_speech_audio = random.choice(list(speech_sources.items()))
    noise_name, selected_noise_audio = random.choice(list(noise_sources.items()))

    # 4. Truncate the selected speech and noise audio files to the shortest length
    min_len_current = min(len(selected_speech_audio), len(selected_noise_audio))
    speech_to_use_current = selected_speech_audio[:min_len_current]
    noise_to_use_current = selected_noise_audio[:min_len_current]

    # 5. Call the custom_generate_room_mix function
    mixed_audio, clean_speech_mic1, clean_speech_mic2, clean_noise_mic1, clean_noise_mic2 = \
        custom_room_mix.custom_generate_room_mix(
            speech_to_use_current, noise_to_use_current, sr=sr, plot=False
        )

    # Ensure output audio is normalized to prevent clipping if sum exceeds 1.0 or -1.0
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 1.0:
        mixed_audio = mixed_audio / max_val
    max_val_s1 = np.max(np.abs(clean_speech_mic1))
    if max_val_s1 > 1.0:
        clean_speech_mic1 = clean_speech_mic1 / max_val_s1
    max_val_s2 = np.max(np.abs(clean_speech_mic2))
    if max_val_s2 > 1.0:
        clean_speech_mic2 = clean_speech_mic2 / max_val_s2
    max_val_n1 = np.max(np.abs(clean_noise_mic1))
    if max_val_n1 > 1.0:
        clean_noise_mic1 = clean_noise_mic1 / max_val_n1
    max_val_n2 = np.max(np.abs(clean_noise_mic2))
    if max_val_n2 > 1.0:
        clean_noise_mic2 = clean_noise_mic2 / max_val_n2

    # 6. Save each generated audio file
    mix_id = f"mix_{i+1:03d}"

    # Save mixed audio (2 channels)
    mixed_audio_filepath = os.path.join(mixed_audio_dir, f"{mix_id}.wav")
    sf.write(mixed_audio_filepath, mixed_audio, sr)
    print(f"Saved mixed audio: {mixed_audio_filepath}")

    # Save clean speech for each mic
    clean_speech_mic1_filepath = os.path.join(clean_speech_dir, f"{mix_id}_speech_mic1.wav")
    sf.write(clean_speech_mic1_filepath, clean_speech_mic1, sr)
    print(f"Saved clean speech mic 1: {clean_speech_mic1_filepath}")

    clean_speech_mic2_filepath = os.path.join(clean_speech_dir, f"{mix_id}_speech_mic2.wav")
    sf.write(clean_speech_mic2_filepath, clean_speech_mic2, sr)
    print(f"Saved clean speech mic 2: {clean_speech_mic2_filepath}")

    # Save clean noise for each mic
    clean_noise_mic1_filepath = os.path.join(clean_noise_dir, f"{mix_id}_noise_mic1.wav")
    sf.write(clean_noise_mic1_filepath, clean_noise_mic1, sr)
    print(f"Saved clean noise mic 1: {clean_noise_mic1_filepath}")

    clean_noise_mic2_filepath = os.path.join(clean_noise_dir, f"{mix_id}_noise_mic2.wav")
    sf.write(clean_noise_mic2_filepath, clean_noise_mic2, sr)
    print(f"Saved clean noise mic 2: {clean_noise_mic2_filepath}")

print(f"\nSuccessfully generated {num_generations} mixed audio files and their components.")
