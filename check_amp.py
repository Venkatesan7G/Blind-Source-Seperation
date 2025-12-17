import soundfile as sf
import numpy as np
from audiomentations import AddGaussianNoise

my_waveform_ndarray, sample_rate = sf.read("speech_recordings/speechcn.wav")
print(f"Original sample rate: {sample_rate} Hz")