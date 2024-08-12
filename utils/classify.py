import os
import pydicom
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define the directory containing your DICOM files
input_directory = '../../our_knee/knee_dicom_sample'
output_directory = '../../our_knee/knee_classified_sample'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Helper function to convert DICOM pixel data to PNG
def dicom_to_png(dicom_file, output_file):
    dataset = pydicom.dcmread(dicom_file)
    pixel_array = dataset.pixel_array
    plt.imshow(pixel_array, cmap='gray')
    plt.axis('off')  # Hide axis
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()


# Iterate through all DICOM files in the input directory
for filename in os.listdir(input_directory):
    dicom_file = os.path.join(input_directory, filename)
    
    try:
        # Read the DICOM file
        dataset = pydicom.dcmread(dicom_file)
        # print(dataset)
        # Extract the Study Description


        study_description = dataset.get((0x0008, 0x1030), 'Unknown description')
        if study_description != 'Unknown description':
            study_description = study_description.value  # Clean up the description
        # print(study_description,type(study_description))
        if study_description == "XR KNEE EXTERNAL ARCHIVE ONLY":
            # with open("class.txt", 'a') as file:
            #     file.write(filename+study_description+"\n")
            continue

        series = dataset.get((0x0008, 0x103e), 'Unknown series')
        if series != 'Unknown series':
            series = series.value  # Clean up the description.

        protocol = dataset.get((0x0018, 0x1030), 'Unknown protocol')
        if protocol != 'Unknown protocol':
            protocol = protocol.value  # Clean up the description

        extra = dataset.get((0x0018, 0x1400), 'Unknown extra')
        if extra != 'Unknown extra':
            extra = extra.value


        if ("protocol" in series.lower() or "protocol" in protocol.lower() or "protocol" in extra.lower()):
            continue
        elif (("sun" in series.lower() or "sun" in protocol.lower() or "sun" in extra.lower()) or \
            ("pat" in series.lower() or "pat" in protocol.lower() or "pat" in extra.lower())) and \
            ("bi" in series.lower() or "bi" in protocol.lower() or "bi" in extra.lower()):
            view = "bisun"
        elif (("sun" in series.lower() or "sun" in protocol.lower() or "sun" in extra.lower()) or \
            ("pat" in series.lower() or "pat" in protocol.lower() or "pat" in extra.lower())) and \
            (not ("bi" in series.lower() or "bi" in protocol.lower() or "bi" in extra.lower())):
            view = "sun"
        elif (("ap" in series.lower() or "ap" in protocol.lower() or "ap" in extra.lower()) or \
            ("pa" in series.lower() or "pa" in protocol.lower() or "pa" in extra.lower())) and \
            ("bi" in series.lower() or "bi" in protocol.lower() or "bi" in extra.lower()):
            view = "biap"
        elif (("ap" in series.lower() or "ap" in protocol.lower() or "ap" in extra.lower()) or \
            ("pa" in series.lower() or "pa" in protocol.lower() or "pa" in extra.lower())) and \
            (not ("bi" in series.lower() or "bi" in protocol.lower() or "bi" in extra.lower())):
            view = "ap"
        elif ("lat" in series.lower() or "lat" in protocol.lower() or "lat" in extra.lower()) and \
            (not ("bi" in series.lower() or "bi" in protocol.lower() or "bi" in extra.lower())):
            view = "lat"
        elif ("tun" in series.lower() or "tun" in protocol.lower() or "tun" in extra.lower()):
            view = "tun"
        elif ("obl" in series.lower() or "obl" in protocol.lower() or "obl" in extra.lower()):
            view = "obl"
        else:
            with open("../../our_knee/knee_classified_sample/class.txt", 'a') as file:
                file.write(filename+study_description+series+protocol+"\n")
            continue


        # if ("left" in series.lower() or "left" in protocol.lower() or "left" in extra.lower()):
        #     side = "L"
        # elif ("right" in series.lower() or "right" in protocol.lower() or "right" in extra.lower()):
        #     view = "R"
        # else:
        #     with open("side.txt", 'a') as file:
        #         file.write(filename+study_description+series+protocol+"\n")
        #     continue


        # Create a folder for the Study Description if it doesn't exist
        study_folder = os.path.join(output_directory, view)
        os.makedirs(study_folder, exist_ok=True)

        # Convert the DICOM image to PNG and save it in the respective folder
        png_file = os.path.join(study_folder, f'{os.path.splitext(filename)[0]}.png')
        dicom_to_png(dicom_file, png_file)

        print(f'Processed {filename} and saved as {png_file}')

    except Exception as e:
        print(f'Failed to process {filename}: {e}')
        with open("../../our_knee/knee_classified_sample/failure.txt", 'a') as file:
                file.write(filename+study_description+series+protocol+str(e)+"\n")

