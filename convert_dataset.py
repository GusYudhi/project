import os

# Fungsi untuk memastikan direktori ada, jika tidak, buatlah
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

for phase in ['datasets/train', 'datasets/test']:
    # Buat folder images dan labels jika belum ada
    ensure_dir(os.path.join(phase, 'images'))
    ensure_dir(os.path.join(phase, 'labels'))

    # Loop through all folders except images and labels
    for folder in os.listdir(phase):
        folder_path = os.path.join(phase, folder)
        if os.path.isdir(folder_path) and folder not in ['images', 'labels']:
            # Loop through all files in the folder
            for file in os.listdir(folder_path):
                src_path = os.path.join(folder_path, file)
                if os.path.isfile(src_path):
                    if file.endswith('.txt'):
                        dst_path = os.path.join(phase, 'labels', file)
                        print(f'Moving {src_path} to {dst_path}')
                        # Move to labels folder
                        os.rename(src_path, dst_path)
                    else:
                        dst_path = os.path.join(phase, 'images', file)
                        print(f'Moving {src_path} to {dst_path}')
                        # Move image file to {phase}/images folder
                        os.rename(src_path, dst_path)
        else:
            print(f'Skipping {folder_path}')

print('done')
