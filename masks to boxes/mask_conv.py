import os
import json
import cv2
import numpy as np

# Function to find the bounding box coordinates
def find_bounding_box(mask):
    # Find the indices of non-zero (white) pixels in the mask
    non_zero_indices = np.argwhere(mask > 0)
    
    if len(non_zero_indices) == 0:
        # If there are no non-zero pixels, return None
        return None
    
    # Extract x and y coordinates of non-zero pixels
    y_coordinates, x_coordinates = non_zero_indices[:, 0], non_zero_indices[:, 1]
    
    # Calculate the bounding box coordinates
    x_min = int(np.min(x_coordinates))
    y_min = int(np.min(y_coordinates))
    x_max = int(np.max(x_coordinates))
    y_max = int(np.max(y_coordinates))
    
    return [x_min, y_min, x_max, y_max]  # Convert to Python list

# Function to load images and masks along with filenames
def load_images_and_masks(images_dir, masks_dir, grayscale=False):
    images = []
    masks = []
    filenames = []  # Store filenames to match masks with their corresponding images
    
    for image_file in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_file)
        mask_file = os.path.splitext(image_file)[0] + "_mask.png"
        mask_path = os.path.join(masks_dir, mask_file)
        
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert the image to grayscale if required
        if grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        images.append(image)
        masks.append(mask)
        filenames.append(image_file)
    
    images = np.array(images)
    masks = np.array(masks)
    
    return images, masks, filenames

# Specify the directories for annotated images and masks
images_dir = "D:\\python\\meow\\pruby\\studia\\m2 bot\\masks to boxes\\not_labeled_stones\\"
masks_dir = "D:\\python\\meow\\pruby\\studia\\m2 bot\\masks to boxes\\masks\\"

# Load images, masks, and get image filenames
images, masks, image_filenames = load_images_and_masks(images_dir, masks_dir, grayscale=True)

# Find bounding box coordinates for each mask
bounding_boxes = []
for mask in masks:
    bbox = find_bounding_box(mask)
    bounding_boxes.append(bbox)

# Create a dictionary to store filenames and bounding box coordinates
data = {}
for filename, bbox in zip(image_filenames, bounding_boxes):
    data[filename] = bbox

# Save the dictionary to a JSON file in a different directory
json_filename = "D:\\python\\meow\\pruby\\studia\\m2 bot\\masks to boxes\\annotations.json"
with open(json_filename, "w") as json_file:
    json.dump(data, json_file)

print("Annotations saved to:", json_filename)