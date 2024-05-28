import csv
import os
import json
from pathlib import Path

from PIL import Image


# Function to rotate a point around a center
def rotate_point(point, center, angle):
    """Rotate a point around a center by a given angle (in degrees)."""
    import math
    angle_rad = math.radians(angle)
    x, y = point
    cx, cy = center
    new_x = cx + (x - cx) * math.cos(angle_rad) - (y - cy) * math.sin(angle_rad)
    new_y = cy + (x - cx) * math.sin(angle_rad) + (y - cy) * math.cos(angle_rad)
    return int(new_x), int(new_y)


# Define the data directory
version = 'v2'
base_dir = 'C:\\Users\\Faraz\\PycharmProjects\\File\\data\\OCR'
label_dir = f'C:\\Users\\Faraz\\PycharmProjects\\File\\data\\OCR\\label\\{version}.json'

Path(os.path.join(base_dir, 'processed', version)).mkdir(parents=True, exist_ok=True)

# Load JSON data
with open(label_dir, 'r', encoding='utf-8') as f:
    data = json.load(f)

image_data = []
i = 0
# Process each entry in the JSON data
for entry in data:
    image_name = entry['ocr'].split('/')[-1].split('-')[-1]  # Extract image name
    image_path = os.path.join(base_dir, 'card', image_name)

    # Load the original image
    original_image = Image.open(image_path)

    # Calculate the rotation matrix if the image has rotation
    if 'image_rotation' in entry:
        rotation_angle = entry['image_rotation']
        original_image = original_image.rotate(rotation_angle, expand=True)

    # Process each bounding box and corresponding transcription
    for j, (bbox, transcription) in enumerate(zip(entry['bbox'], entry['transcription'])):
        i += 1
        # Convert percentage to pixels
        x = int(bbox['x'] * original_image.width / 100)
        y = int(bbox['y'] * original_image.height / 100)
        width = int(bbox['width'] * original_image.width / 100)
        height = int(bbox['height'] * original_image.height / 100)

        # Rotate bounding box if there's rotation
        if 'rotation' in bbox:
            bbox_rotation = bbox['rotation']
            # Convert rotation to negative as PIL rotates counter-clockwise
            bbox_rotation = -bbox_rotation
            # Rotate the bounding box
            bbox_center = (x + width / 2, y + height / 2)
            bbox_points = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
            bbox_rotated_points = [rotate_point(point, bbox_center, bbox_rotation) for point in bbox_points]
            # Get new bounding box coordinates
            x = min(point[0] for point in bbox_rotated_points)
            y = min(point[1] for point in bbox_rotated_points)
            width = max(point[0] for point in bbox_rotated_points) - x
            height = max(point[1] for point in bbox_rotated_points) - y

        # Crop the image
        cropped_image = original_image.crop((x, y, x + width, y + height))

        # Resize cropped image to 600x400
        cropped_image = cropped_image.resize((256, 64))

        # Save resized image with transcription name
        # save_path = os.path.join(base_dir, 'processed', 'v2', f"{transcription}.jpg")
        save_path = os.path.join(base_dir, 'processed', version, f"{i}.jpg")
        cropped_image.save(save_path)

        # image_data.append((os.path.abspath(save_path), transcription))
        image_data.append((f"{i}.jpg", str(transcription)))

print("Image cropping, resizing, and saving completed.")

csv_file_path = os.path.join(base_dir, 'processed', version, 'image_data.csv')

with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(['image_path', 'text'])
    # Write rows
    for image_path, text in image_data:
        writer.writerow([image_path, text])

print(f"CSV file saved at: {csv_file_path}")

# import pandas as pd
# # Define CSV file path
# csv_file_path = os.path.join(data_dir, 'processed', 'image_data.csv')
#
# # Convert image data to DataFrame
# df = pd.DataFrame(image_data, columns=['image_path', 'text'])
#
# # Save DataFrame to CSV file with utf-8 encoding
# df.to_csv(csv_file_path, index=False)
# # df.to_json(csv_file_path, index=False)
#
# print(f"CSV file saved at: {csv_file_path}")