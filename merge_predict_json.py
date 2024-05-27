import json
import glob
import sys
# Get a list of all JSON files in the current directory
json_files = glob.glob(f'{sys.argv[1]}*.json')

# Create an empty dictionary to store the merged data
merged_data = {}

# Iterate over each JSON file
for file in json_files:
    # Open the file and load the JSON data
    with open(file, 'r') as f:
        data = json.load(f)
        print(list(data.keys())) 
    # Merge the data into the merged_data dictionary
    merged_data.update(data)
    merged_data = dict(sorted([[key, value] for key, value in merged_data.items()], key=lambda x: int(x[0]), reverse=False))
    
# Write the merged data to a new JSON file
with open('predict_dev.json', 'w') as f:
    json.dump(merged_data, f, indent=4)
