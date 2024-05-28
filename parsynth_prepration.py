import os
import csv

# Define the data directory
dataset_dir = 'C:\\Users\\Faraz\\PycharmProjects\\File\\data\\OCR\\data\\processed\\train'
output_csv = 'C:\\Users\\Faraz\\PycharmProjects\\File\\data\\OCR\\data\\processed\\train\\image_data.txt'

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

image_data = []

# Process each image in the dataset directory
for filename in os.listdir(dataset_dir):
    if filename.endswith('.jpg'):
        # Split the filename to extract the label
        parts = filename.split('_')
        if len(parts) == 2:
            index, label_with_ext = parts
            label = label_with_ext.rsplit('.', 1)[0]  # Remove the file extension from the label
            image_path = os.path.join(dataset_dir, filename)
            image_data.append((filename, label))
            # image_data.append((image_path, label))

# Write the data to a CSV file
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(['image_path', 'text'])
    # Write rows
    for image_path, text in image_data:
        writer.writerow([image_path, text])

print(f"CSV file saved at: {output_csv}")
