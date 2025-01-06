# import json
# import pydicom

# # Function to filter JSON entries
# def filter_dicom_json(json_file_path, output_file_path):
#     with open(json_file_path, 'r') as f:
#         data = json.load(f)
    
#     filtered_data = []
    
#     for entry in data:
#         img_path = entry['img_path']
#         try:
#             # Read the DICOM file
#             dataset = pydicom.dcmread(img_path)
#             # Get the study description
#             # print(dataset)
#             # break
#             series_description = dataset.get((0x0008, 0x103e), 'Unknown View')
#             # Check if "AP" is in the description
#             # print(study_description)
#             # study_description = string(study_description)
#             # print(study_description)
#             if "pa" in str(series_description).lower() or "ap" in str(series_description).lower():
#                 filtered_data.append(entry)
#         except Exception as e:
#             print(f"Error processing {img_path}: {e}")
    
#     # Save the filtered data to a new JSON file
#     with open(output_file_path, 'w') as f:
#         json.dump(filtered_data, f, indent=4)
#     print(f"Filtered JSON saved to {output_file_path}")

# # Input and output file paths
# input_json = './data/ourdata_whole/train.json'  # Replace with your input JSON file path
# output_json = './data/ourdata/train.json'  # Replace with your desired output file path

# # Run the filter function
# filter_dicom_json(input_json, output_json)



# ###going to extract accession number
# import json

# # Function to extract unique accession numbers from a JSON file
# def extract_accession_numbers(file_path):
#     with open(file_path, "r") as file:
#         data = json.load(file)
    
#     # Extract and return accession numbers as a set
#     accession_numbers = {
#         item["img_path"].split("/images/")[1].split("/")[0] for item in data
#     }
#     return accession_numbers

# # Input JSON files
# json_file_1 = "./data/ourdata/train.json"
# json_file_2 = "./data/ourdata/test.json"

# # Extract unique accession numbers from both files
# accession_numbers_1 = extract_accession_numbers(json_file_1)
# accession_numbers_2 = extract_accession_numbers(json_file_2)

# # Combine and avoid duplicates using set union
# unique_accession_numbers = accession_numbers_1.union(accession_numbers_2)

# # Save to file
# output_file = "unique_accession_numbers.txt"
# with open(output_file, "w") as file:
#     for number in sorted(unique_accession_numbers):  # Sorting for consistency
#         file.write(number + "\n")

# print(f"Unique accession numbers saved to {output_file}")


# with open(output_file, "r") as file:
#     print(file.read())


###add an accession line

# import json

# # Function to add accession number to each image entry
# def add_accession_number(file_path, output_path):
#     with open(file_path, "r") as file:
#         data = json.load(file)

#     # Add accession number to each entry
#     for item in data:
#         img_path = item["img_path"]
#         accession_number = img_path.split("/images/")[1].split("/")[0]
#         item["accession"] = accession_number  # Add accession number to the item

#     # Save the updated JSON to the output file
#     with open(output_path, "w") as file:
#         json.dump(data, file, indent=4)

#     print(f"Updated JSON saved to {output_path}")

# # Input and output file paths
# input_file = "./data/ourdata/test.json"  # Replace with your input file path
# output_file = "./newtest.json"  # Output file with accession numbers

# # Call the function to update the JSON
# add_accession_number(input_file, output_file)


### add the pathology column

import pandas as pd
import csv
# Define the pathologies
pathologies = [
    "fracture",
    "osteoarthritis",
    "joint effusion",
    "healing/healed fracture",
    "soft tissue swelling",
    "orif",
    "arthroplasty",
    "enthesopathy",
    "intra-articular fracture",
    "heterotopic ossification",
    "chondrocalcinosis",
    "osteochondral injury",
    "intraarticular body",
    "osteotomy",
    "No findings"
]


def append_to_csv(file_name, data):
    file_exists = False
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            file_exists = True
    except FileNotFoundError:
        pass

    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Accession Number", "Pathologies"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)

# # Read the CSV file
# data = pd.read_csv("predictions.csv")
# reader = csv.reader(data)
# rows = [row for row in reader]
# print(rows)

with open('predictions.csv','r',encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
print(rows)

kk = 1
output_file = "accession_pathologies.csv"
for row in rows:

    if kk == 1:
        kk+=1
        continue
    # Extract accession number and pathology flags
    accession = row[0]
    flags = row[1:]
    print(flags, type(flags))

    # Identify the present pathologies
    present_pathologies = [pathologies[i] for i, flag in enumerate(flags) if float(flag) > 0.0]

    # Prepare the output data
    output_data = [
        {"Accession Number": accession, "Pathologies": ", ".join(present_pathologies)}
    ]
    # Append to CSV
    append_to_csv(output_file, output_data)
    

# # Write to CSV
# output_file = "accession_pathologies.csv"
# with open(output_file, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.DictWriter(file, fieldnames=["Accession Number", "Pathologies"])
#     writer.writeheader()
#     writer.writerows(output_data)

# print(f"CSV file '{output_file}' has been created with the accession and pathologies.")

