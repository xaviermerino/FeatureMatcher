import os
import numpy as np

# Directory where the files will be created
output_dir = "./generated_files"

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Total number of files to generate
total_files = 10000

# Number of subjects
num_subjects = 50

# Number of pictures per subject
pictures_per_subject = total_files // num_subjects

# Starting subject ID (5-digit numbers)
start_subject_id = 10000
end_subject_id = start_subject_id + num_subjects - 1

# Loop over subjects
for subject_id in range(start_subject_id, end_subject_id + 1):
    # Loop over pictures per subject
    for picture_num in range(1, pictures_per_subject + 1):
        # Generate a random digit between 1 and 5 for num3
        num3 = np.random.randint(1, 6)  # Random integer between 1 and 5

        # Construct the filename
        filename = f"{subject_id}_{picture_num}_{num3}.npy"

        # Generate a 512-element array with random numbers
        data = np.random.rand(512)

        # Save the array to a .npy file
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, data)

print(f"Generated {total_files} files in '{output_dir}'.")
