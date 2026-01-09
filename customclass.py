import tensorflow as tf
import numpy as np
import os

class SpeechSeparationDataGenerator(tf.keras.utils.Sequence):
    """
    Custom Data Generator for Speech Separation using STFT magnitude spectrograms.
    Generates batches of mixed audio magnitudes (X) and clean speech magnitudes (y).
    """

    def __init__(self, data_paths, n_bins, target_n_bins, n_frames_example, batch_size, shuffle=True):
        self.data_paths = data_paths
        self.n_bins = n_bins  # Original number of frequency bins (e.g., 257)
        self.target_n_bins = target_n_bins # Model's output frequency bins (e.g., 256)
        self.n_frames_example = n_frames_example # Fixed number of time frames
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data_paths))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.data_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs for the current batch
        batch_data_paths = [self.data_paths[k] for k in indexes]
        ## added
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # Generate data
        X, y = self.__data_generation(batch_data_paths)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _process_spectrogram(self, spec, is_mixed=False):
        """
        Pads/truncates spectrograms to n_frames_example and crops frequency bins.
        Input spec shape: (n_bins, n_frames) for mono, or (n_channels, n_bins, n_frames) for mixed.
        Output spec shape: (target_n_bins, n_frames_example, 1) for mono, or list of two of these for mixed.
        """
        processed_specs = []

        if is_mixed:
            # For mixed audio, input spec is (n_channels, n_bins, n_frames)
            mic1_spec = spec[0, :, :]
            mic2_spec = spec[1, :, :]
            specs_to_process = [mic1_spec, mic2_spec]
        else:
            # For clean audio, input spec is (n_bins, n_frames)
            specs_to_process = [spec]

        for current_spec in specs_to_process:
            # 1. Handle padding/truncating time dimension
            original_n_frames = current_spec.shape[1]
            if original_n_frames < self.n_frames_example:
                # Pad with zeros
                padding = np.zeros((current_spec.shape[0], self.n_frames_example - original_n_frames))
                current_spec = np.concatenate((current_spec, padding), axis=1)
            elif original_n_frames > self.n_frames_example:
                # Truncate
                current_spec = current_spec[:, :self.n_frames_example]

            # 2. Crop frequency dimension (from original n_bins to target_n_bins)
            current_spec = current_spec[:self.target_n_bins, :]

            # 3. Reshape (add channel dimension) to (target_n_bins, n_frames_example, 1)
            current_spec = np.expand_dims(current_spec, axis=-1)
            processed_specs.append(current_spec)

        return processed_specs[0] if not is_mixed else processed_specs

    def __data_generation(self, batch_data_paths):
        ##added
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        """Generates data containing batch_size samples."""
        # X_batch will have shape (batch_size, target_n_bins, n_frames_example, 2)
        X_batch = np.empty((self.batch_size, self.target_n_bins, self.n_frames_example, 2))
        # y_batch will have shape (batch_size, target_n_bins, n_frames_example, 2)
        y_batch = np.empty((self.batch_size, self.target_n_bins, self.n_frames_example, 2))

        # Loop through each data item (mix) in the batch
        for i, data_item in enumerate(batch_data_paths):
            # Load and process mixed magnitude spectrograms
            mixed_mag_path = os.path.join(BASE_DIR, data_item['mixed_mag_path'])

            print("Trying to load:", mixed_mag_path)
            print("Exists?", os.path.exists(mixed_mag_path))

            mixed_mag_spec_full = np.load(mixed_mag_path)

            # print("Trying to load:", data_item['mixed_mag_path'])
            # print("Exists?", os.path.exists(data_item['mixed_mag_path']))
            # #adding full path
            # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            # full_path = os.path.join(BASE_DIR, data_item['mixed_mag_path'])
            # mixed_mag_spec_full = np.load(full_path)
            #mixed_mag_spec_full = np.load(data_item['mixed_mag_path'])
            processed_mixed_mic1, processed_mixed_mic2 = self._process_spectrogram(mixed_mag_spec_full, is_mixed=True)

            # Stack processed mixed magnitudes for X_batch
            X_batch[i, :, :, 0] = processed_mixed_mic1[:, :, 0]
            X_batch[i, :, :, 1] = processed_mixed_mic2[:, :, 0]

            # Load and process clean speech magnitude spectrograms for each mic
            clean_speech_mic1_mag_spec = np.load(data_item['clean_speech_mic1_mag_path'])
            processed_clean_speech_mic1 = self._process_spectrogram(clean_speech_mic1_mag_spec, is_mixed=False)

            clean_speech_mic2_mag_spec = np.load(data_item['clean_speech_mic2_mag_path'])
            processed_clean_speech_mic2 = self._process_spectrogram(clean_speech_mic2_mag_spec, is_mixed=False)

            # Stack processed clean speech magnitudes for y_batch
            y_batch[i, :, :, 0] = processed_clean_speech_mic1[:, :, 0]
            y_batch[i, :, :, 1] = processed_clean_speech_mic2[:, :, 0]

        return X_batch, y_batch

print("Custom SpeechSeparationDataGenerator class defined.")