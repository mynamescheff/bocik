import os

input_dir = 'img/not_labeled_stones/cropped/'

# Iterate through files and rename as needed
for i in range(68, 112 + 1):
    old_filename = f'cropped_Screenshot_{i}.png'
    new_filename = f'cropped_Screenshot_{i - 1}.png'
    
    old_path = os.path.join(input_dir, old_filename)
    new_path = os.path.join(input_dir, new_filename)
    
    os.rename(old_path, new_path)

print("Files renamed successfully.")