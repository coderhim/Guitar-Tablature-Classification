import os

def remove_hex_cln_from_filenames(folder_path):
    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)
        
        # Ensure it's a file
        if not os.path.isfile(old_path):
            continue

        # Remove all occurrences of "_hex_cln"
        new_filename = filename.replace("_hex_cln", "")
        new_path = os.path.join(folder_path, new_filename)

        # Rename if the name changes
        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

folder_path = r'D:\Code playground\seminar_audioTab_\audio_hex-pickup_debleeded'
remove_hex_cln_from_filenames(folder_path)
