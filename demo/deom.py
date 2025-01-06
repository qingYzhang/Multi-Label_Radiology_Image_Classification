import os
import pydicom
from PIL import Image
import numpy as np

def dicom_to_png(dicom_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over DICOM files in the specified folder
    for dicom_file in os.listdir(dicom_folder):
        if dicom_file.endswith(".dcm"):
            dicom_path = os.path.join(dicom_folder, dicom_file)
            
            # Read the DICOM file
            dicom_data = pydicom.dcmread(dicom_path)
            
            # Convert pixel data to an image
            image_data = dicom_data.pixel_array
            
            # Normalize image data (if necessary) to 8-bit range (0-255)
            image_data = np.uint8(image_data / np.max(image_data) * 255)
            
            # Convert the numpy array to a Pillow image
            image = Image.fromarray(image_data)
            
            # Save the image as PNG
            output_path = os.path.join(output_folder, dicom_file.replace(".dcm", ".png"))
            image.save(output_path)
            print(f"Saved {output_path}")

# Example usage
dicom_folder = "./"
output_folder = "./"
dicom_to_png(dicom_folder, output_folder)
