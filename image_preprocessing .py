import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
import os

def preprocess_images(images_dir, masks_dir, cropped_dir, target_size=(256, 256), batch_size=32, validation_split=0.2):
    input_data = []
    output_data = []

    # Loop through all images in the directory
    for img_name in os.listdir(images_dir):
        if img_name.endswith('.png'):
            img_path = os.path.join(images_dir, img_name)
            mask_path = os.path.join(masks_dir, img_name.replace('.png', '_mask.png'))
            cropped_path = os.path.join(cropped_dir, f'cropped_{img_name}')

            # Load the original image, mask, and cropped image
            img = cv.imread(img_path)
            mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
            cropped_img = cv.imread(cropped_path)

            # Preprocessing steps
            resized_img = cv.resize(img, target_size)
            normalized_img = resized_img / 255.0  # Normalize to [0, 1]
            output_data.append(cv.resize(mask, target_size))  # Resize the mask

            # Append the normalized image to the input data
            input_data.append(normalized_img)

    # Split data into training and validation sets
    train_input, val_input, train_output, val_output = train_test_split(input_data, output_data, test_size=validation_split, random_state=42)

    # Create batches
    num_train_batches = len(train_input) // batch_size
    num_val_batches = len(val_input) // batch_size

    for batch_idx in range(num_train_batches):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size

        batch_input = np.array(train_input[batch_start:batch_end])
        batch_output = np.array(train_output[batch_start:batch_end])

        yield batch_input, batch_output

    for batch_idx in range(num_val_batches):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size

        batch_input = np.array(val_input[batch_start:batch_end])
        batch_output = np.array(val_output[batch_start:batch_end])

        yield batch_input, batch_output