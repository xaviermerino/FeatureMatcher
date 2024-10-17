#!/bin/bash

# Directory where the files will be created
output_dir="./generated_files"

# Create the directory if it doesn't exist
mkdir -p "$output_dir"

# Total number of files to generate
total_files=100

# Number of subjects
num_subjects=10

# Number of pictures per subject
pictures_per_subject=$((total_files / num_subjects))

# Starting subject ID (5-digit numbers)
start_subject_id=10000
end_subject_id=$((start_subject_id + num_subjects - 1))

# Loop over subjects
for subject_id in $(seq $start_subject_id $end_subject_id)
do
    # Loop over pictures per subject
    for picture_num in $(seq 1 $pictures_per_subject)
    do
        # Generate a random digit between 1 and 5 for num3
        num3=$(shuf -i 1-5 -n 1)

        # Construct the filename
        filename="${subject_id}_${picture_num}_${num3}.npy"

        # Create the empty file
        touch "$output_dir/$filename"
    done
done

echo "Generated $total_files files in '$output_dir'."
