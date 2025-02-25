import os
import numpy as np
import librosa

def process_all_audio(dataset_path, window_size=0.2, hop_size=0.1, save_path='output'):
    # Ensure the output directory exists
    os.makedirs(save_path, exist_ok=True)

    # Function to remove low-energy noise
    def cqt_lim(CQT):
        new_CQT = np.copy(CQT)
        new_CQT[new_CQT < -60] = -120
        return new_CQT

    # Get all audio files in the dataset directory
    audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]

    # Iterate through all audio files
    for audio_file in audio_files:
        audio_path = os.path.join(dataset_path, audio_file)

        # Load audio
        data, sr = librosa.load(audio_path, sr=None, mono=True)

        # Parameters for sliding window
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)

        # Ensure valid segments only (drop incomplete ones)
        num_segments = (len(data) - window_samples) // hop_samples + 1

        print(f'Processing {num_segments} valid segments for: {audio_file}')

        segment_count = 0

        for i in range(num_segments):
            # Extract segment
            start_sample = i * hop_samples
            end_sample = start_sample + window_samples

            # Drop segments smaller than 0.2s
            if end_sample > len(data):
                break

            segment = data[start_sample:end_sample]

            # Skip segments smaller than 0.2 seconds
            if len(segment) < window_samples:
                continue

            # Dynamically adjust fmin to avoid warnings
            min_freq = librosa.note_to_hz('C1') if len(segment) >= 256 else None

            # Compute CQT
            CQT = librosa.cqt(segment, sr=sr, hop_length=1024, n_bins=96, bins_per_octave=12, fmin=min_freq)
            CQT_mag = np.abs(CQT)**4
            CQTdB = librosa.amplitude_to_db(CQT_mag, ref=np.amax)
            new_CQT = cqt_lim(CQTdB)

            # Save segment as .npy (preserving the original name and segment index)
            base_name = os.path.splitext(audio_file)[0]
            npy_filename = f"{base_name}_segment_{segment_count}.npy"
            np.save(os.path.join(save_path, npy_filename), new_CQT)

            segment_count += 1

        print(f'Saved {segment_count} valid segments for {audio_file} in {save_path}')

dataset_path = r'D:\Code playground\seminar_audioTab_\audio'
save_path = r'D:\Code playground\seminar_audioTab_\cqt_audio'

process_all_audio(dataset_path, save_path=save_path)