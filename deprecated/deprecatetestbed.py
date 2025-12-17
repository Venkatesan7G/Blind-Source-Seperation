import numpy as np
import torch
import torchaudio
from torchaudio.pipelines import CONVTASNET_BASE_LIBRI2MIX
import pyroomacoustics as pra
import mir_eval.separation
import time
import pandas as pd  # For table output

# Load model
bundle = CONVTASNET_BASE_LIBRI2MIX
model = bundle.get_model()
sample_rate = bundle.sample_rate  # 8000 Hz

# Function to generate a simulated mixture (2 sources, 1 mic for mono; extend to multi-mic)
def generate_mixture(duration=5.0):
    # Create a shoebox room (4x6x3m)
    room_dim = [4, 6, 3]
    room = pra.ShoeBox(room_dim, fs=sample_rate, max_order=3)  # Low reverb for simplicity

    # Add 2 sources with random positions and signals (white noise for testing; replace with speech)
    source_positions = [[1, 2, 1.5], [3, 4, 1.0]]
    clean_sources = [np.random.randn(int(sample_rate * duration)) for _ in range(2)]  # Clean refs

    for pos, sig in zip(source_positions, clean_sources):
        room.add_source(pos, signal=sig)

    # Add a single mic (for mono; for stereo, add array with 2 positions)
    mic_pos = np.array([[2.0, 3.0, 1.5]])  # Shape [1, 3] for 1 mic
    room.add_microphone_array(mic_pos.T)

    # Simulate
    room.simulate()

    # Mixed signal (mono: [1, 1, samples] for batch=1, channel=1)
    mix = torch.from_numpy(room.mic_array.signals[0]).unsqueeze(0).unsqueeze(0).float()

    # Clean sources as numpy [2, samples]
    clean = np.stack(clean_sources)

    return mix, clean

# Testbed: Run on N mixtures
N = 10  # Number of test mixtures
sirs, sars, proc_times = [], [], []

for i in range(N):
    mix, clean_ref = generate_mixture()

    # Separate and time
    start = time.time()
    with torch.no_grad():
        separated = model(mix)  # [1, 2, samples]
        separated = separated.squeeze(0)  # [2, samples] - Fix batch dim
    proc_time = time.time() - start
    proc_times.append(proc_time)

    # To numpy for metrics
    separated_np = separated.numpy()  # (2, samples)

    # Align lengths (simulation may pad mix/output)
    min_len = min(clean_ref.shape[1], separated_np.shape[1])
    clean_ref = clean_ref[:, :min_len]
    separated_np = separated_np[:, :min_len]

    # Compute BSS metrics (average over sources)
    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(clean_ref, separated_np)
    sirs.append(np.mean(sir))
    sars.append(np.mean(sar))

# Compute stats
data = {
    'Metric': ['SIR', 'SAR', 'Proc. Time (s)'],
    'Mean': [np.mean(sirs), np.mean(sars), np.mean(proc_times)],
    'Std. Dev.': [np.std(sirs), np.std(sars), np.std(proc_times)]
}

df = pd.DataFrame(data)
print(df.to_string(index=False))

# For your table format (assuming stereo, but here mono; adjust mic setup)
print("\nMic Setup | SIR Mean | SIR Std. Dev. | SAR Mean | SAR Std. Dev. | Proc. Time Mean (s) | Proc. Time Std. Dev. | Your Model")
print(f"mono      | {np.mean(sirs):.1f}     | {np.std(sirs):.1f}          | {np.mean(sars):.1f}     | {np.std(sars):.1f}          | {np.mean(proc_times):.2f}                 | {np.std(proc_times):.2f}                   | ConvTasNet")