import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

def audio_CQT_parallel(file_num, start, dur=0.2):  # start and dur in seconds

    # Paths and loading audio
    path = r'D:\Code playground\seminar_audioTab_\audio_hex-pickup_debleeded'
    audio_files = os.listdir(path)
    audio_path = os.path.join(path, audio_files[file_num])
    audio_name = os.path.splitext(audio_files[file_num])[0]
    # Function to limit noise
    def cqt_lim(CQT):
        new_CQT = np.copy(CQT)
        new_CQT[new_CQT < -60] = -120
        return new_CQT

    # Load audio
    data, sr = librosa.load(audio_path, sr=None, mono=True, offset=start, duration=dur)

    # Perform the Constant-Q Transform
    CQT = librosa.cqt(data, sr=44100, hop_length=1024, fmin=None, n_bins=96, bins_per_octave=12)
    CQT_mag = librosa.magphase(CQT)[0]**4
    CQTdB = librosa.core.amplitude_to_db(CQT_mag, ref = np.amax)

    # Apply noise limit
    new_CQT = cqt_lim(CQTdB)

    # Save CQT as grayscale image
    output_dir = "cqt_images"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(new_CQT, sr=44100, hop_length=1024, x_axis='time', y_axis='cqt_note')
    plt.axis('off')

    output_path = os.path.join(output_dir, f"{audio_name}_segment_{file_num}_{start:.2f}.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Saved: {output_path}")

def process_all_files_parallel(start, dur=0.2, max_images=45000):
    path = r'D:\Code playground\seminar_audioTab_\audio_hex-pickup_debleeded'
    audio_files = os.listdir(path)

    total_files = len(audio_files)
    num_images_per_file = max_images // total_files

    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(total_files):
            for j in range(num_images_per_file):
                offset = start + j * dur
                futures.append(executor.submit(audio_CQT_parallel, i, offset, dur))

        for future in futures:
            future.result()

# Example usage

def main():
    process_all_files_parallel(start=0, dur=0.2, max_images=45000)

if __name__ == "__main__":
    main()