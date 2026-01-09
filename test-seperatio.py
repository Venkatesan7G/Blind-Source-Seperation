import numpy as np
import tensorflow as tf
from scipy.io import wavfile
import librosa

# Load the pre-trained model
model = tf.keras.models.load_model('best_model.keras')

# Function to calculate SIR (Signal-to-Interference Ratio)
def calculate_sir(original_signal, separated_signal):
    # Ensure both signals are the same length for SIR calculation
    min_length = min(len(original_signal), len(separated_signal))
    original_signal = original_signal[:min_length]
    separated_signal = separated_signal[:min_length]

    signal_power = np.sum(original_signal ** 2)
    interference_power = np.sum((separated_signal - original_signal) ** 2)
    if interference_power == 0:
        return np.inf  # Perfect separation
    sir = 10 * np.log10(signal_power / interference_power)
    return sir

# Function to calculate SAR (Signal-to-Artifacts Ratio)
def calculate_sar(original_signal, separated_signal):
    # Ensure both signals are the same length for SAR calculation
    min_length = min(len(original_signal), len(separated_signal))
    original_signal = original_signal[:min_length]
    separated_signal = separated_signal[:min_length]

    signal_power = np.sum(original_signal ** 2)
    artifact_power = np.sum((separated_signal - original_signal) ** 2)
    if artifact_power == 0:
        return np.inf  # Perfect separation
    sar = 10 * np.log10(signal_power / artifact_power)
    return sar

# Load the original clean sources
sr1, original_audio_1 = wavfile.read("og1.wav")
sr2, original_audio_2 = wavfile.read("og2.wav")

# Load the mixed audio file
mixed_audio_path = 'test01.wav'
sr_mixed, mixed_audio = wavfile.read(mixed_audio_path)

# Convert to mono if stereo
if len(mixed_audio.shape) > 1:
    mixed_audio = np.mean(mixed_audio, axis=1)

# Preprocess the mixed audio into magnitude spectrogram
n_fft = 512       # Use the same n_fft as in the model training
hop_length = 128  # Use the same hop_length as in the model training

D = librosa.stft(mixed_audio, n_fft=n_fft, hop_length=hop_length)
magnitude, phase = librosa.magphase(D)

# Pad or crop the magnitude spectrogram to match the input shape the model expects
# The model expects (256, 620) size, so we crop or pad accordingly
# Ensure it has 256 frequency bins (we crop or pad in the frequency dimension)
magnitude = magnitude[:256, :]

# Ensure it has 620 frames (crop or pad in the time dimension)
target_frames = 620
if magnitude.shape[1] < target_frames:
    pad_width = target_frames - magnitude.shape[1]
    magnitude = np.pad(magnitude, ((0, 0), (0, pad_width)), mode='constant')
elif magnitude.shape[1] > target_frames:
    magnitude = magnitude[:, :target_frames]

# Stack the magnitude spectrogram into two channels for stereo input (as required by the model)
magnitude_stereo = np.stack([magnitude, magnitude], axis=-1)
magnitude_stereo = np.expand_dims(magnitude_stereo, axis=0)  # Add batch dimension

# Predict the separated sources using the model
separated_sources = model.predict(magnitude_stereo)

# Post-process: Convert magnitude back to waveform
source_1_mag = separated_sources[0, :, :, 0]  # First separated source
source_2_mag = separated_sources[0, :, :, 1]  # Second separated source

# Match the phase dimensions to the magnitude's dimensions (256, 620)
# First, crop or pad the phase to match the magnitude dimensions
phase = phase[:256, :]

# If phase is smaller than 620 frames, pad it; if larger, crop it
if phase.shape[1] < 620:
    phase = np.pad(phase, ((0, 0), (0, 620 - phase.shape[1])), mode='constant')
elif phase.shape[1] > 620:
    phase = phase[:, :620]

# Reconstruct the complex spectrogram for each source
source_1_complex = source_1_mag * np.exp(1j * phase)
source_2_complex = source_2_mag * np.exp(1j * phase)

# Inverse STFT to get the time-domain signals
source_1_audio = librosa.istft(source_1_complex)
source_2_audio = librosa.istft(source_2_complex)

# Ensure that the lengths of the separated signals match the original signals
# Find the shortest length between the original and separated signals
min_len_1 = min(len(original_audio_1), len(source_1_audio))
min_len_2 = min(len(original_audio_2), len(source_2_audio))

# Trim to the shortest length to avoid dimension mismatch
source_1_audio = source_1_audio[:min_len_1]
source_2_audio = source_2_audio[:min_len_2]

# Save the separated audio signals
wavfile.write("source_1_separated.wav", sr_mixed, source_1_audio.astype(np.float32))
wavfile.write("source_2_separated.wav", sr_mixed, source_2_audio.astype(np.float32))

# Calculate SIR and SAR for all combinations
sir_1_1 = calculate_sir(original_audio_1, source_1_audio)
sar_1_1 = calculate_sar(original_audio_1, source_1_audio)

sir_1_2 = calculate_sir(original_audio_2, source_1_audio)
sar_1_2 = calculate_sar(original_audio_2, source_1_audio)

sir_2_1 = calculate_sir(original_audio_1, source_2_audio)
sar_2_1 = calculate_sar(original_audio_1, source_2_audio)

sir_2_2 = calculate_sir(original_audio_2, source_2_audio)
sar_2_2 = calculate_sar(original_audio_2, source_2_audio)

# Compare and print the best results
best_sir_1 = max(sir_1_1, sir_1_2)
best_sar_1 = max(sar_1_1, sar_1_2)

best_sir_2 = max(sir_2_1, sir_2_2)
best_sar_2 = max(sar_2_1, sar_2_2)

print(f"Best SIR for Source 1: {best_sir_1:.2f} dB")
print(f"Best SAR for Source 1: {best_sar_1:.2f} dB")

print(f"Best SIR for Source 2: {best_sir_2:.2f} dB")
print(f"Best SAR for Source 2: {best_sar_2:.2f} dB")
