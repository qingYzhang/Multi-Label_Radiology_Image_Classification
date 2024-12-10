import json
import pydicom

# Function to filter JSON entries
def filter_dicom_json(json_file_path, output_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    filtered_data = []
    
    for entry in data:
        img_path = entry['img_path']
        try:
            # Read the DICOM file
            dataset = pydicom.dcmread(img_path)
            # Get the study description
            # print(dataset)
            # break
            series_description = dataset.get((0x0008, 0x103e), 'Unknown View')
            # Check if "AP" is in the description
            # print(study_description)
            # study_description = string(study_description)
            # print(study_description)
            if "pa" in str(series_description).lower() or "ap" in str(series_description).lower():
                filtered_data.append(entry)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Save the filtered data to a new JSON file
    with open(output_file_path, 'w') as f:
        json.dump(filtered_data, f, indent=4)
    print(f"Filtered JSON saved to {output_file_path}")

# Input and output file paths
input_json = './data/ourdata_whole/train.json'  # Replace with your input JSON file path
output_json = './data/ourdata/train.json'  # Replace with your desired output file path

# Run the filter function
filter_dicom_json(input_json, output_json)