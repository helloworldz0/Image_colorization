import cv2
import os
import glob

# Set the source and destination folders
source_folder = './Data/Coloured_Face_Images/'
destination_folder = './Data/Grayscale_Face_Images/'

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Get list of image files (you can adjust the extensions)
image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(source_folder, ext)))

# Process each image
for img_path in image_paths:
    # Extract the filename
    filename = os.path.basename(img_path)

    # Read the image
    color_img = cv2.imread(img_path)
    if color_img is None:
        print(f"Failed to load image: {filename}")
        continue

    # Convert to grayscale
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # Save to destination folder with same filename
    output_path = os.path.join(destination_folder, filename)
    cv2.imwrite(output_path, gray_img)

    print(f"Converted and saved: {output_path}")

print("All images processed.")
