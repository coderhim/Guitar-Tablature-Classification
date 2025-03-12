import os
import concurrent.futures

def get_files_in_folder(folder):
    """Return a set of filenames in a folder."""
    return set(os.listdir(folder))

def delete_extra_files(folder, files_in_other_folder):
    """Delete files from a folder that aren't present in the other folder."""
    for filename in os.listdir(folder):
        if filename not in files_in_other_folder:
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted {file_path}")

def sync_folders(folder1, folder2):
    """Synchronize two folders by deleting extra files."""
    files_in_folder1 = get_files_in_folder(folder1)
    files_in_folder2 = get_files_in_folder(folder2)

    # Create thread pool for parallel execution
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks to delete extra files in both folders
        future1 = executor.submit(delete_extra_files, folder1, files_in_folder2)
        future2 = executor.submit(delete_extra_files, folder2, files_in_folder1)
        
        # Wait for both operations to complete
        future1.result()
        future2.result()

    print("Folder synchronization complete.")

# Example usage
folder1 = r'D:\Code playground\seminar_audioTab_\cqt_images'
folder2 = r'D:\Code playground\seminar_audioTab_\tablature_segments'

sync_folders(folder1, folder2)
