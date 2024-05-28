import os
import json
import random
import glob
import uuid

# Specify the directory containing your JSON files
# Initialize an empty dictionary to store sampled data
result_data = {}
# Function to sample keys from a JSON file
def sample_keys(json_data, num_keys=5):
    return random.sample(json_data.keys(), min(num_keys, len(json_data)))

json_files = glob.glob("exp_version3/**/*.json", recursive=True)
# Ignore "mix_5_all.json"
# Iterate through each file in the directory
for filename in json_files:
    if "mix" in filename":
        continue
    if filename.endswith('.json'):
        print(filename)
        # Read the JSON file
        with open(filename, 'r') as file:
            json_data = json.load(file)

        # Sample keys from the JSON file
        sampled_keys = sample_keys(json_data)

        # Store sampled keys in the result_data dictionary
        for key in sampled_keys:
            result_data[str(uuid.uuid4())] = json_data[key] 

# Write the result_data to a new JSON file
output_file_path = 'mix_5_all.json'
with open(output_file_path, 'w') as output_file:
    print("Num keys", len(result_data.keys()), len(json_files))
    json.dump(result_data, output_file)

print(f"Sampling complete. Result stored in {output_file_path}")