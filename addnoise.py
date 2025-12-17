import soundfile as sf
import numpy as np
from audiomentations import AddGaussianNoise, PolarityInversion

my_waveform_ndarray, sample_rate = sf.read("speech_recordings/speechcn.wav")

my_waveform_ndarray = my_waveform_ndarray.astype(np.float32).T if my_waveform_ndarray.ndim > 1 else my_waveform_ndarray.astype(np.float32)

transform = AddGaussianNoise(
    min_amplitude=0.005,
    max_amplitude=0.015,
    p=1.0
)

augmented_sound = transform(my_waveform_ndarray, sample_rate=48000)
if augmented_sound.ndim > 1:
    augmented_sound = augmented_sound.T
sf.write("bgsound.wav", augmented_sound, 48000)





# import soundfile as sf
# import numpy as np
# from audiomentations import AddBackgroundNoise, PolarityInversion

# my_waveform_ndarray, sample_rate = sf.read("speech_recordings/speechcn.wav")
# my_waveform_ndarray = my_waveform_ndarray.astype(np.float32).T if my_waveform_ndarray.ndim > 1 else my_waveform_ndarray.astype(np.float32)
# transform = AddBackgroundNoise(
#     sounds_path="speech_recordings",
#     min_snr_db=3.0,
#     max_snr_db=30.0,
#     noise_transform=PolarityInversion(),
#     p=1.0
# )

# augmented_sound = transform(my_waveform_ndarray, sample_rate=16000)
# sf.write("bgsound.wav", augmented_sound, 16000) 