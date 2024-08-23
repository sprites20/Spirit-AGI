from huggingface_hub import HfApi, login, upload_file
import pandas as pd
import h5py
import numpy as np
import os
from datasets import Dataset, DatasetDict

# Your Hugging Face token
HF_TOKEN = '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'  # Replace with your actual Hugging Face token

# Authenticate with Hugging Face
login(token=HF_TOKEN)

# Define the generate_prompt function
def generate_prompt(data_point):
    """Generate input text based on a prompt, task instruction, (context info.), and answer

    :param data_point: dict: Data point
    :return: str: tokenized prompt
    """
    prefix_text = 'Below is an instruction that describes a task. Write a response that ' \
                   'appropriately completes the request.\n\n'
    if data_point['input']:
        text = f"""<start_of_turn>user {prefix_text} {data_point["instruction"]} here are the inputs {data_point["input"]} <end_of_turn>\n<start_of_turn>model{data_point["output"]} <end_of_turn>"""
    else:
        text = f"""<start_of_turn>user {prefix_text} {data_point["instruction"]} <end_of_turn>\n<start_of_turn>model{data_point["output"]} <end_of_turn>"""
    return text

# Existing DataFrame
new_data = pd.DataFrame({
    'prompt': []
})

# Define the data_point for each entry
data_points = [
    {"input": "", "instruction": "Provide a number", "output": "3"},
    {"input": "", "instruction": "Provide a number", "output": "4"}
]

# Generate prompts and create a new DataFrame
generated_prompts = [generate_prompt(dp) for dp in data_points]
generated_data = pd.DataFrame({'prompt': generated_prompts})

# Append the new DataFrame to the existing one
new_data = pd.concat([new_data, generated_data], ignore_index=True)

#print(new_data)

# Convert DataFrame to numpy array of bytes for text data
new_data_array = new_data['prompt'].astype('S256').to_numpy()

# File path
hdf5_file = 'prompts.h5'

# Function to append data to HDF5
def append_data_to_hdf5(file_path, new_data):
    with h5py.File(file_path, 'a') as f:
        if 'prompts' in f:
            # Dataset exists, extend it
            dataset = f['prompts']
            # Resize dataset to accommodate new data
            old_size = dataset.shape[0]
            new_size = old_size + new_data.shape[0]
            dataset.resize(new_size, axis=0)
            # Append new data
            dataset[old_size:] = new_data
        else:
            # Dataset does not exist, create it
            f.create_dataset('prompts', data=new_data, maxshape=(None,), dtype='S256')

# Append new data
append_data_to_hdf5(hdf5_file, new_data_array)

# Hugging Face repository setup
repo_id = "IanVilla/gemma-2-finetuning"  # Replace with your Hugging Face username and desired repo name

# Initialize Hugging Face API
api = HfApi()

# Create the repository if it does not exist
try:
    # Check if repo exists
    repo_info = api.repo_info(repo_id)
    print(f"Repository '{repo_id}' already exists.")
except Exception as e:
    # Create repo if it does not exist
    try:
        api.create_repo(repo_id, repo_type="dataset")
        print(f"Repository '{repo_id}' created.")
    except Exception as e:
        print(f"An error occurred while creating the repository: {e}")

# Upload the file to Hugging Face (this will overwrite the existing file)
try:
    upload_file(
        path_or_fileobj=hdf5_file,
        path_in_repo=hdf5_file,
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message="Updated HDF5 file with new data",
    )
    print(f"File '{hdf5_file}' updated in repo '{repo_id}'.")
except Exception as e:
    print(f"An error occurred while uploading the file: {e}")
    
def read_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        if 'prompts' in f:
            # Access the dataset
            dataset = f['prompts']
            # Convert to DataFrame
            # Since the data is stored as bytes, decode and convert to DataFrame
            data = [entry.decode('utf-8') for entry in dataset[:]]
            df = pd.DataFrame(data, columns=['prompt'])
            return df
        else:
            print("Dataset 'prompts' not found.")
            return None

# Read data from HDF5 file
df = read_hdf5('prompts.h5')
print(df)

# Convert the DataFrame to a Dataset object
dataset = Dataset.from_pandas(df)

# Create a DatasetDict object
dataset_dict = DatasetDict({
    "train": dataset
})

# Upload the dataset to Hugging Face
dataset_dict.push_to_hub(repo_id=repo_id, token=HF_TOKEN)


from datasets import load_dataset

# Force download
dataset = load_dataset("IanVilla/gemma-2-finetuning", cache_dir=None)

print(dataset)


dataset = dataset["train"].to_pandas()
dataset = dataset.train_test_split(test_size=0.1)
train_data = dataset["train"]
test_data = dataset["test"]
print(train_data)
print(test_data)
