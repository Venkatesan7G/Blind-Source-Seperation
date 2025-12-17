import torch
import torchaudio
from torchaudio.pipelines import CONVTASNET_BASE_LIBRI2MIX

# Load the pretrained bundle
bundle = CONVTASNET_BASE_LIBRI2MIX
model = bundle.get_model()
sample_rate = bundle.sample_rate  # Typically 8000 Hz

# Load a mixed audio file (assume mono; resample if needed)
mix_waveform, sr = torchaudio.load('path/to/your_mixed_audio.wav')
if sr != sample_rate:
    resampler = torchaudio.transforms.Resample(sr, sample_rate)
    mix_waveform = resampler(mix_waveform)
if mix_waveform.shape[0] > 1:  # If multichannel, average to mono for now
    mix_waveform = mix_waveform.mean(dim=0, keepdim=True)

# Perform separation (with timing)
import time
start_time = time.time()
with torch.no_grad():
    separated_waveforms = model(mix_waveform)  # Output: tensor [num_sources, time]
proc_time = time.time() - start_time
print(f"Processing time: {proc_time:.2f} seconds")

# Save separated waves
for i, sep in enumerate(separated_waveforms):
    torchaudio.save(f'separated_source_{i}.wav', sep.unsqueeze(0), sample_rate)