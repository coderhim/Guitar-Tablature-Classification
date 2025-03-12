import os
import numpy as np
import jams
import librosa

def extract_midi_from_jams(jam, start, stop):
    midi_data = []

    for annotation in jam.annotations:
        if annotation.namespace == 'note_midi':
            data = [(obs.time, obs.duration, obs.value) for obs in annotation]
            midi_data.extend(data)

    if not midi_data:
        return np.array([], dtype=np.int32)

    midi_data = np.atleast_2d(np.array(midi_data, dtype=np.float32))[:, :3]

    midi_end_times = midi_data[:, 0] + midi_data[:, 1]

    tab_1 = np.where((midi_data[:, 0] >= start) & (midi_data[:, 0] <= stop))[0]
    tab_2 = np.where((midi_end_times >= start) & (midi_end_times <= stop))[0]
    tab_3 = np.where((midi_data[:, 0] < start) & (midi_end_times > stop))[0]

    tab_ind = np.union1d(np.union1d(tab_1, tab_2), tab_3)

    if tab_ind.size > 0:
        MIDI_val = np.unique(midi_data[tab_ind, 2].round().astype(int))
        if MIDI_val.size >= 6:
            MIDI_val = MIDI_val[:6]
    else:
        MIDI_val = np.array([], dtype=np.int32)

    return MIDI_val

def notes_to_fret(MIDI_val):
    Fret = np.array([40, 45, 50, 55, 59, 64])[:, np.newaxis] + np.arange(18)
    f_row = np.full((len(MIDI_val), 6), -1, dtype=np.int32)
    f_col = np.full((len(MIDI_val), 6), -1, dtype=np.int32)

    for fcnt, note in enumerate(MIDI_val):
        rows, cols = np.where(Fret == note)
        
        # Safely handle the number of matches
        num_matches = min(len(rows), 6)
        f_row[fcnt, :num_matches] = rows[:num_matches]
        f_col[fcnt, :num_matches] = cols[:num_matches]

    return f_row, f_col

def optimum_strings(f_row, f_col):
    if not np.any(f_col >= 0):
        return np.zeros((6, 18), dtype=np.int32)

    # Find all valid positions for the first note
    valid_first_positions = np.where(f_col[0] >= 0)[0]
    
    if len(valid_first_positions) == 0:
        return np.zeros((6, 18), dtype=np.int32)
    
    f_solutions = []
    
    # Create a solution candidate for each valid position of the first note
    for pri in valid_first_positions:
        f_sol = np.copy(f_col)
        
        # For each subsequent note, calculate distance from the first note's position
        for sub_r in range(1, min(len(f_row), 6)):
            valid_indices = (f_col[sub_r] >= 0)
            if np.any(valid_indices):
                f_sol[sub_r, valid_indices] = np.abs(f_col[0, pri] - f_col[sub_r, valid_indices])

        # Mark invalid positions
        f_sol[f_sol < 0] = 999
        f_solutions.append(f_sol)

    # Calculate the minimum solution for each candidate
    min_solutions = []
    for f_sol in f_solutions:
        min_sol = np.zeros(min(len(f_row), 6), dtype=np.int32)
        for i in range(min(len(f_row), 6)):
            valid_min = np.min(f_sol[i][f_sol[i] != 999]) if np.any(f_sol[i] != 999) else 0
            min_sol[i] = valid_min
        min_solutions.append(min_sol)

    # Find the best solution (minimizing the sum of distances)
    min_sums = [np.sum(min_sol) for min_sol in min_solutions]
    best_solution_idx = np.argmin(min_sums) if min_sums else 0
    
    return f_solutions[best_solution_idx] if f_solutions else np.zeros((6, 18), dtype=np.int32)

def create_tablature(f_row, f_col):
    True_tab = np.zeros((6, 18), dtype=np.int32)

    for i in range(len(f_row)):
        for j in range(min(6, f_row.shape[1])):
            if (f_row[i, j] >= 0 and f_row[i, j] < 6 and 
                f_col[i, j] >= 0 and f_col[i, j] < 18):
                True_tab[5 - f_row[i, j], f_col[i, j]] = 1

    return True_tab

def jams_to_tablature(jams_file, start=0, stop=0.2):
    try:
        jam = jams.load(jams_file)
        MIDI_val = extract_midi_from_jams(jam, start, stop)

        if MIDI_val.size == 0:
            return np.ones((6, 19), dtype=np.int32)  # All rows indicate no note

        f_row, f_col = notes_to_fret(MIDI_val)
        optimal_solution = optimum_strings(f_row, f_col)

        # Initialize output with 1s in the first column (no note) and zeros elsewhere
        tab_matrix = np.zeros((6, 19), dtype=np.int32)
        tab_matrix[:, 0] = 1

        if isinstance(optimal_solution, np.ndarray) and optimal_solution.size > 0:
            raw_tab = create_tablature(f_row, optimal_solution)

            # For each string (row), check if a note exists and update the first column
            for string in range(6):
                if np.any(raw_tab[string] > 0):
                    tab_matrix[string, 0] = 0  # Note exists
                tab_matrix[string, 1:] = raw_tab[string]

        return tab_matrix

    except Exception as e:
        print(f"Error processing {jams_file} at time {start}-{stop}: {e}")
        return np.ones((6, 19), dtype=np.int32)  # Return matrix indicating no note


# def cqt_lim(CQT):
#     new_CQT = np.copy(CQT)
#     new_CQT[new_CQT < -60] = -120
#     return new_CQT

# def process_aligned_dataset(audio_folder, jams_folder, output_folder, window_size=0.2, hop_size=0.1):
#     """
#     Process audio files and jams files together, creating aligned segments
#     """
#     # Create output folders
#     cqt_output = os.path.join(output_folder, 'cqt_features')
#     tab_output = os.path.join(output_folder, 'tablatures')
    
#     os.makedirs(cqt_output, exist_ok=True)
#     os.makedirs(tab_output, exist_ok=True)
    
#     # Map JAMS files to their corresponding audio files
#     jams_files = {os.path.splitext(f)[0]: os.path.join(jams_folder, f) 
#                  for f in os.listdir(jams_folder) if f.endswith('.jams')}
    
#     # Process each audio file
#     for audio_file in os.listdir(audio_folder):
#         if not audio_file.endswith('.wav'):
#             continue
            
#         base_name = os.path.splitext(audio_file)[0]
#         audio_path = os.path.join(audio_folder, audio_file)
        
#         # Find corresponding JAMS file
#         jams_path = None
#         for jams_key in jams_files:
#             if jams_key in base_name:  # Match partial names
#                 jams_path = jams_files[jams_key]
#                 break
                
#         if not jams_path:
#             print(f"No matching JAMS file found for {audio_file}, skipping")
#             continue
            
#         print(f"Processing {audio_file} with annotation {os.path.basename(jams_path)}")
        
#         # Load audio
#         data, sr = librosa.load(audio_path, sr=None, mono=True)
        
#         # Parameters for sliding window
#         window_samples = int(window_size * sr)
#         hop_samples = int(hop_size * sr)
        
#         # Calculate number of valid segments
#         num_segments = (len(data) - window_samples) // hop_samples + 1
        
#         print(f"Creating {num_segments} segments")
        
#         # Process each segment
#         for i in range(num_segments):
#             # Calculate segment time boundaries
#             start_time = i * hop_size
#             end_time = start_time + window_size
            
#             # Extract audio segment
#             start_sample = i * hop_samples
#             end_sample = start_sample + window_samples
            
#             if end_sample > len(data):
#                 break
                
#             segment = data[start_sample:end_sample]
            
#             # Skip segments smaller than window size
#             if len(segment) < window_samples:
#                 continue
                
#             # Compute CQT for audio segment
#             min_freq = librosa.note_to_hz('C1') if len(segment) >= 256 else None
#             CQT = librosa.cqt(segment, sr=sr, hop_length=1024, n_bins=96, bins_per_octave=12, fmin=min_freq)
#             CQT_mag = np.abs(CQT)**4
#             CQTdB = librosa.amplitude_to_db(CQT_mag, ref=np.amax)
#             new_CQT = cqt_lim(CQTdB)
            
#             # Generate tablature for the same time segment
#             tablature = jams_to_tablature(jams_path, start=start_time, stop=end_time)
            
#             # Save both files with matching names
#             segment_name = f"{base_name}_segment_{i}"
#             cqt_filename = os.path.join(cqt_output, f"{segment_name}.npy")
#             tab_filename = os.path.join(tab_output, f"{segment_name}.npy")
            
#             np.save(cqt_filename, new_CQT)
#             np.save(tab_filename, tablature)
            
#         print(f"Finished processing {audio_file}")

# def align_existing_segments(audio_segments_folder, jams_folder, tab_output_folder):
#     """
#     Alternative approach if you already have audio segments and need to create matching tablatures
#     """
#     os.makedirs(tab_output_folder, exist_ok=True)
    
#     # Load all JAMS files
#     jams_files = {os.path.splitext(os.path.basename(f))[0]: os.path.join(jams_folder, f) 
#                  for f in os.listdir(jams_folder) if f.endswith('.jams')}
    
#     # Group segment files by original audio file
#     segment_groups = {}
    
#     for segment_file in os.listdir(audio_segments_folder):
#         if not segment_file.endswith('.npy'):
#             continue
            
#         # Parse segment information from filename
#         parts = os.path.splitext(segment_file)[0].split('_segment_')
#         if len(parts) != 2:
#             continue
            
#         base_name = parts[0]
#         segment_num = int(parts[1])
        
#         if base_name not in segment_groups:
#             segment_groups[base_name] = []
            
#         segment_groups[base_name].append((segment_num, segment_file))
    
#     # Process each group
#     for base_name, segments in segment_groups.items():
#         # Find matching JAMS file
#         jams_path = None
#         for jams_key in jams_files:
#             if jams_key in base_name:  # Match partial names
#                 jams_path = jams_files[jams_key]
#                 break
                
#         if not jams_path:
#             print(f"No matching JAMS file found for {base_name}, skipping")
#             continue
            
#         print(f"Processing {len(segments)} segments for {base_name}")
        
#         # Sort segments by number
#         segments.sort(key=lambda x: x[0])
        
#         # Process each segment
#         for segment_num, segment_file in segments:
#             # Calculate segment time
#             start_time = segment_num * 0.1  # Assuming hop_size=0.1
#             end_time = start_time + 0.2     # Assuming window_size=0.2
            
#             # Generate tablature for this time segment
#             tablature = jams_to_tablature(jams_path, start=start_time, stop=end_time)
            
#             # Save tablature with matching name
#             tab_filename = os.path.join(tab_output_folder, segment_file)
#             np.save(tab_filename, tablature)
            
#         print(f"Finished processing {base_name}")
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_segment(segment_file, base_name, jams_path, tab_output_folder):
    """
    Process a single audio segment and generate corresponding tablature.
    """
    try:
        # Extract segment number from filename
        segment_num = int(segment_file.split('_segment_')[1].split('.')[0])
        
        # Calculate segment time (Assuming hop_size=0.1, window_size=0.2)
        start_time = segment_num * 0.1
        end_time = start_time + 0.2
        
        # Generate tablature for this segment
        tablature = jams_to_tablature(jams_path, start=start_time, stop=end_time)
        
        # Save the tablature
        tab_filename = os.path.join(tab_output_folder, segment_file)
        np.save(tab_filename, tablature)

        return f"Processed {segment_file} for {base_name}"
    
    except Exception as e:
        return f"Error processing {segment_file}: {e}"

def align_existing_segments(audio_segments_folder, jams_folder, tab_output_folder):
    """
    Parallelized approach for aligning audio segments with JAMS files to create tablatures.
    """
    os.makedirs(tab_output_folder, exist_ok=True)
    
    # Load all JAMS files (keyed by base filename)
    jams_files = {
        os.path.splitext(os.path.basename(f))[0]: os.path.join(jams_folder, f)
        for f in os.listdir(jams_folder) if f.endswith('.jams')
    }
    
    # Group segment files by original audio file
    segment_groups = {}
    for segment_file in os.listdir(audio_segments_folder):
        if not segment_file.endswith('.png'):
            continue
        
        # Parse segment information
        parts = os.path.splitext(segment_file)[0].split('_segment_')
        if len(parts) != 2:
            continue
        
        base_name = parts[0]
        segment_groups.setdefault(base_name, []).append(segment_file)
    
    # Process each group in parallel
    with ProcessPoolExecutor() as executor:
        futures = []
        
        for base_name, segments in segment_groups.items():
            # Find matching JAMS file (partial match)
            jams_path = next((path for key, path in jams_files.items() if key in base_name), None)
            if not jams_path:
                print(f"No matching JAMS file for {base_name}, skipping.")
                continue
            
            print(f"Processing {len(segments)} segments for {base_name} in parallel.")
            
            # Submit each segment for processing
            for segment_file in segments:
                futures.append(executor.submit(process_segment, segment_file, base_name, jams_path, tab_output_folder))
        
        # Collect results
        for future in as_completed(futures):
            print(future.result())

    print("Finished processing all segments.")

# Example usage
if __name__ == '__main__':
    # Option 1: Process everything from scratch
    # audio_folder = r"D:\Code playground\seminar_audioTab_\audio"
    # jams_folder = r"D:\Code playground\seminar_audioTab_\annotation"
    # output_folder = r"D:\Code playground\seminar_audioTab_\processed_data"
    
    # process_aligned_dataset(audio_folder, jams_folder, output_folder)
    
    # Option 2: If you already have CQT segments and just need matching tablatures
    audio_segments_folder = r"D:\Code playground\seminar_audioTab_\cqt_images"
    jams_folder = r"D:\Code playground\seminar_audioTab_\annotation"
    tab_output_folder = r"D:\Code playground\seminar_audioTab_\tablature_segments"
    # audio_segments_folder = r"D:\Code playground\seminar_audioTab_\cqt_images"
    # jams_folder = r"D:\Code playground\seminar_audioTab_\annotation"
    # tab_output_folder = r"D:\Code playground\seminar_audioTab_\tablature_segments"
    
    align_existing_segments(audio_segments_folder, jams_folder, tab_output_folder)