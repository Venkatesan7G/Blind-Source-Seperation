import librosa
import librosa.display
import os
import numpy as np
import soundfile as sf

# STFT Parameters
n_fft = 512       # Window size
hop_length = 128  # Hop length (stride)

# Create directories for STFT data
stft_output_dir = 'stft_data'
mag_mixed_dir = os.path.join(stft_output_dir, 'magnitude', 'mixed_audio')
phase_mixed_dir = os.path.join(stft_output_dir, 'phase', 'mixed_audio')
mag_clean_speech_dir = os.path.join(stft_output_dir, 'magnitude', 'clean_speech')
phase_clean_speech_dir = os.path.join(stft_output_dir, 'phase', 'clean_speech')
mag_clean_noise_dir = os.path.join(stft_output_dir, 'magnitude', 'clean_noise')
phase_clean_noise_dir = os.path.join(stft_output_dir, 'phase', 'clean_noise')

os.makedirs(mag_mixed_dir, exist_ok=True)
os.makedirs(phase_mixed_dir, exist_ok=True)
os.makedirs(mag_clean_speech_dir, exist_ok=True)
os.makedirs(phase_clean_speech_dir, exist_ok=True)
os.makedirs(mag_clean_noise_dir, exist_ok=True)
os.makedirs(phase_clean_noise_dir, exist_ok=True)

print("Created STFT output directories.")

def apply_stft_and_save(audio_filepath, sr, output_mag_dir, output_phase_dir, n_fft, hop_length):
    audio, _ = sf.read(audio_filepath)

    # Handle multi-channel audio for mixed files
    if audio.ndim > 1:
        stft_channels = []
        for i in range(audio.shape[1]):
            stft_channels.append(librosa.stft(audio[:, i], n_fft=n_fft, hop_length=hop_length))
        stft_result = np.array(stft_channels) # Shape (n_channels, n_bins, n_frames)
    else:
        stft_result = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

    magnitude = np.abs(stft_result)
    phase = np.angle(stft_result)

    base_filename = os.path.basename(audio_filepath).replace('.wav', '')

    # Save magnitude
    mag_filepath = os.path.join(output_mag_dir, f"{base_filename}_mag.npy")
    np.save(mag_filepath, magnitude)

    # Save phase
    phase_filepath = os.path.join(output_phase_dir, f"{base_filename}_phase.npy")
    np.save(phase_filepath, phase)

    return mag_filepath, phase_filepath

# Iterate through all generated WAV files and apply STFT
root_output_dir = 'generated_dataset'
mixed_audio_source_dir = os.path.join(root_output_dir, 'mixed_audio')
clean_speech_source_dir = os.path.join(root_output_dir, 'clean_speech')
clean_noise_source_dir = os.path.join(root_output_dir, 'clean_noise')

all_mixed_files = sorted([f for f in os.listdir(mixed_audio_source_dir) if f.endswith('.wav')])

for mix_file in all_mixed_files:
    mix_id = mix_file.replace('.wav', '')

    print(f"Processing {mix_id} for STFT...")

    # Process mixed audio
    mixed_audio_path = os.path.join(mixed_audio_source_dir, mix_file)
    apply_stft_and_save(mixed_audio_path, 16000, mag_mixed_dir, phase_mixed_dir, n_fft, hop_length)

    # Process clean speech for mic1 and mic2
    speech_mic1_file = f"{mix_id}_speech_mic1.wav"
    speech_mic2_file = f"{mix_id}_speech_mic2.wav"
    speech_mic1_path = os.path.join(clean_speech_source_dir, speech_mic1_file)
    speech_mic2_path = os.path.join(clean_speech_source_dir, speech_mic2_file)
    apply_stft_and_save(speech_mic1_path, 16000, mag_clean_speech_dir, phase_clean_speech_dir, n_fft, hop_length)
    apply_stft_and_save(speech_mic2_path, 16000, mag_clean_speech_dir, phase_clean_speech_dir, n_fft, hop_length)

    # Process clean noise for mic1 and mic2
    noise_mic1_file = f"{mix_id}_noise_mic1.wav"
    noise_mic2_file = f"{mix_id}_noise_mic2.wav"
    noise_mic1_path = os.path.join(clean_noise_source_dir, noise_mic1_file)
    noise_mic2_path = os.path.join(clean_noise_source_dir, noise_mic2_file)
    apply_stft_and_save(noise_mic1_path, 16000, mag_clean_noise_dir, phase_clean_noise_dir, n_fft, hop_length)
    apply_stft_and_save(noise_mic2_path, 16000, mag_clean_noise_dir, phase_clean_noise_dir, n_fft, hop_length)

print("STFT processing complete for all generated audio files.")