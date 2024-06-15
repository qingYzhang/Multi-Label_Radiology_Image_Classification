import csv
import json

# Define the CSV file path
csv_file_path = './Data_Entry_2017.csv'

# Define the path to the train list text file
train_list_path = './train_val_list.txt'
test_list_path = './test_list.txt'

# Define the list of possible findings
possible_findings = [
    "Atelectasis","Consolidation","Infiltration","Pneumothorax","Edema","Emphysema","Fibrosis",
    "Effusion","Pneumonia","Pleural_thickening","Cardiomegaly","Nodule Mass","Hernia","No Finding"
]

possible_findings = [finding.lower() for finding in possible_findings]


# Read the train and test lists from the text files
with open(train_list_path, 'r') as train_file:
    train_list = [line.strip() for line in train_file]

with open(test_list_path, 'r') as test_file:
    test_list = [line.strip() for line in test_file]

# Create a dictionary to store the data
data = {
    "train": [],
    "test": []
}

# Function to create a target vector
def create_target_vector(findings, possible_findings):
    vector = [0] * len(possible_findings)
    for finding in findings:
        finding = finding.lower()
        for possible_finding in possible_findings:
            if finding in possible_finding:
                vector[possible_findings.index(possible_finding)] = 1
    return vector

# Read the CSV file and populate the data dictionary
with open(csv_file_path, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        findings = row['Finding Labels'].split('|')
        target_vector = create_target_vector(findings, possible_findings)
        img_path = f"../data/all_images/{row['Image Index']}"

        if row['Image Index'] in train_list:
            data["train"].append({"target": target_vector, "img_path": img_path})
        elif row['Image Index'] in test_list:
            test_target_vector = [-1] * len(possible_findings)
            for i, value in enumerate(target_vector):
                if value == 1:
                    test_target_vector[i] = 1
            data["test"].append({"target": test_target_vector, "img_path": img_path})

# Save the train and test data to JSON files
with open('train_data.json', 'w') as train_file:
    json.dump(data["train"], train_file, indent=4)

with open('test_data.json', 'w') as test_file:
    json.dump(data["test"], test_file, indent=4)

print("JSON files for training and testing data have been created successfully.")
