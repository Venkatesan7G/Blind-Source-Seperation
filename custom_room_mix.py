import pyroomacoustics as pra
import numpy as np
import random
import warnings

def custom_generate_room_mix(speech_audio, noise_audio, sr=16000,
                           room_dim_range=((5.0, 5.0, 2.5), (10.0, 10.0, 3.5)),
                           mic_dist_min=2.0, src_dist_min=1.0,
                           rt60_range=(0.1, 0.5), plot=False):
    """
    Generates a room acoustic mix with two microphones and two sources (speech, noise).

    Args:
        speech_audio (np.ndarray): NumPy array of speech audio samples.
        noise_audio (np.ndarray): NumPy array of noise audio samples.
        sr (int): Sampling rate.
        room_dim_range (tuple): Tuple of (min_dims, max_dims) for room dimensions (x, y, z).
        mic_dist_min (float): Minimum distance between microphones in meters.
        src_dist_min (float): Minimum distance between sources in meters.
        rt60_range (tuple): Tuple of (min_rt60, max_rt60) for room reverberation time.
        plot (bool): If True, plots the room setup.

    Returns:
        tuple: (mixed_audio, clean_speech_mic1, clean_speech_mic2,
                clean_noise_mic1, clean_noise_mic2) as numpy arrays.
    """

    # 1. Randomly select room dimensions and RT60 with retry for valid absorption coefficient
    max_room_param_attempts = 100
    room = None
    for _ in range(max_room_param_attempts):
        min_dims, max_dims = room_dim_range
        room_dim = np.array([
            random.uniform(min_dims[0], max_dims[0]),
            random.uniform(min_dims[1], max_dims[1]),
            random.uniform(min_dims[2], max_dims[2]),
        ])
        rt60 = random.uniform(rt60_range[0], rt60_range[1])

        try:
            e_absorption_raw = pra.inverse_sabine(rt60, room_dim)
            if isinstance(e_absorption_raw, tuple):
                e_absorption = e_absorption_raw[0]
            else:
                e_absorption = e_absorption_raw

            # Ensure absorption coefficient is within valid range [0, 1) after extraction
            if 0 < e_absorption < 1:
                room = pra.ShoeBox(room_dim, fs=sr, materials=pra.Material(e_absorption))
                break # Valid room created, exit retry loop
            else:
                warnings.warn(f"Generated e_absorption {e_absorption:.2f} out of [0, 1) range. Retrying.")
        except ValueError as e:
            warnings.warn(f"ValueError during room parameter generation: {e}. Retrying.")

    if room is None:
        raise RuntimeError("Failed to create a valid room with acceptable parameters after multiple attempts.")

    # Define buffer distance from walls for microphone and source placement
    wall_buffer = 0.5 # meters

    def get_random_pos_in_room(dims, buffer):
        return np.array([random.uniform(buffer, d - buffer) for d in dims])

    # 2. Microphone Placement (at least mic_dist_min apart)
    max_mic_placement_attempts = 100
    mic1_pos = get_random_pos_in_room(room_dim, wall_buffer)
    mic2_pos = None

    for _ in range(max_mic_placement_attempts):
        temp_mic2_pos = get_random_pos_in_room(room_dim, wall_buffer)
        if np.linalg.norm(mic1_pos - temp_mic2_pos) >= mic_dist_min:
            mic2_pos = temp_mic2_pos
            break

    if mic2_pos is None:
        warnings.warn("Could not place second microphone meeting distance criteria after multiple attempts. "
                      "Consider adjusting mic_dist_min or room_dim_range.")
        mic2_pos = mic1_pos + np.array([mic_dist_min, 0, 0]) # Fallback: try to place it along x-axis
        # Ensure fallback position is within room, if not, choose random without min_dist
        if not all(wall_buffer <= c <= d - wall_buffer for c, d in zip(mic2_pos, room_dim)):
            mic2_pos = get_random_pos_in_room(room_dim, wall_buffer)

    mic_positions = np.c_[mic1_pos, mic2_pos]
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs=sr))

    # 3. Source Placement (at least src_dist_min apart from each other, buffered from mics)
    max_src_placement_attempts = 100
    src_buffer_from_mic = 0.3 # Ensure sources are not too close to microphones

    speech_src_pos = get_random_pos_in_room(room_dim, wall_buffer)
    noise_src_pos = None

    mic_positions_list = [mic1_pos, mic2_pos]

    for _ in range(max_src_placement_attempts):
        temp_noise_src_pos = get_random_pos_in_room(room_dim, wall_buffer)
        dist_to_speech = np.linalg.norm(speech_src_pos - temp_noise_src_pos)
        dist_to_mics_ok = True
        for mic_p in mic_positions_list:
            if np.linalg.norm(mic_p - temp_noise_src_pos) < src_buffer_from_mic:
                dist_to_mics_ok = False
                break

        if dist_to_speech >= src_dist_min and dist_to_mics_ok:
            noise_src_pos = temp_noise_src_pos
            break

    if noise_src_pos is None:
        warnings.warn("Could not place second source meeting distance criteria from speech source/mics after multiple attempts. "
                      "Consider adjusting src_dist_min or room_dim_range.")
        noise_src_pos = get_random_pos_in_room(room_dim, wall_buffer)


    # Add sources to the room (signals are stored in source objects)
    _ = room.add_source(speech_src_pos, signal=speech_audio)
    _ = room.add_source(noise_src_pos, signal=noise_audio)

    # Compute RIRs for all sources and microphones
    room.compute_rir()

    # Get RIRs for each source-microphone pair from room.rir
    # room.rir has shape (n_mics, n_sources, rir_length)
    # Assuming speech is source 0 and noise is source 1
    speech_rir_mic1 = room.rir[0][0]
    speech_rir_mic2 = room.rir[1][0]
    noise_rir_mic1 = room.rir[0][1]
    noise_rir_mic2 = room.rir[1][1]

    # Convolve source signals with their respective RIRs to get clean signals at each microphone
    clean_speech_mic1_full = np.convolve(speech_audio, speech_rir_mic1)
    clean_speech_mic2_full = np.convolve(speech_audio, speech_rir_mic2)
    clean_noise_mic1_full = np.convolve(noise_audio, noise_rir_mic1)
    clean_noise_mic2_full = np.convolve(noise_audio, noise_rir_mic2)

    # Determine the minimum length across all full convolved signals
    min_overall_len = min(
        len(clean_speech_mic1_full), len(clean_speech_mic2_full),
        len(clean_noise_mic1_full), len(clean_noise_mic2_full)
    )

    # Truncate all clean signals to the minimum common length
    clean_speech_mic1 = clean_speech_mic1_full[:min_overall_len]
    clean_speech_mic2 = clean_speech_mic2_full[:min_overall_len]
    clean_noise_mic1 = clean_noise_mic1_full[:min_overall_len]
    clean_noise_mic2 = clean_noise_mic2_full[:min_overall_len]

    # Manually create mixed audio by summing the truncated clean signals
    mixed_audio_mic1 = clean_speech_mic1 + clean_noise_mic1
    mixed_audio_mic2 = clean_speech_mic2 + clean_noise_mic2
    mixed_audio = np.c_[mixed_audio_mic1, mixed_audio_mic2] # Shape (n_samples, n_mics)

    # Simulate the room if plotting is requested to prepare for plotting (e.g. `room.plot()`)
    # Note: `room.simulate()` populates `room.mic_array.signals` which is what `room.plot()` expects.
    # If we manually calculate mixed_audio, `room.mic_array.signals` might be empty or incorrect for plotting.
    # For plotting only, we might need to call room.simulate() to fill the signals.
    # However, if we are returning mixed_audio from manual convolution, the `room.simulate()` here
    # will just generate new signals and is not used for return values.
    if plot:
        # Re-run simulation to populate mic_array.signals for plotting. This is slightly redundant
        # if we calculated signals manually, but needed for the plot function's internal logic.
        room.simulate()
        import matplotlib.pyplot as plt
        fig, ax = room.plot()
        ax.set_xlim([0, room_dim[0]])
        ax.set_ylim([0, room_dim[1]])
        ax.set_zlim([0, room_dim[2]])
        plt.title(f'Room Setup (RT60: {rt60:.2f}s)')
        plt.show()

    return mixed_audio, clean_speech_mic1, clean_speech_mic2, clean_noise_mic1, clean_noise_mic2
