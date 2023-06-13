# FeatureMatcher

Feature Matcher for Biometric Templates

Some options available are shown in the table below:

| Option             | Description                                                                                 | Default          | Required |
|--------------------|---------------------------------------------------------------------------------------------|------------------|----------|
| --probe_dir        | Path to the probe directory                                                                 | N/A              | Yes      |
| --gallery_dir      | Path to the gallery directory. If not present, the probe is also the gallery.               | N/A              | No       |
| --output_dir       | Path where results are saved                                                                | N/A              | Yes      |
| --match_type       | Desired matching type (i.e., authentic, impostor, all)                                      | all              | No       |
| --matcher          | Algorithm used to compare features. Only cosine similarity implemented.                     | cosinesimilarity | No       |
| --score_file_type  | Type of score file to output (i.e., txt, csv)                                               | txt              | No       |
| --matrix_file_type | Type of score matrix file to output (i.e., npy, csv)                                        | None             | No       |
| --group_name       | Given name for the comparison or group                                                      | group            | No       |
| --regex_string     | Regex to extract subject ID. If provided, the first group match will be returned as the ID. | None             | No       |


- The `--probe_dir` option provides a path to a directory containing templates in .npy format.
- The `--gallery_dir` option provides a path to a directory containing gallery templates in .npy format. If a path is not provided for gallery directory, this will assume the gallery equals the probe.
- The `--output_dir` option provides the path where the results (scores) are going to be saved.
- The `--match_type` option allows to perform matching and save scores for authentic, impostor, or both.
- The `--group_name` option allows you to set a name for the comparison or grouN.
- The `--score_file_type` option allows you to save the output in a text file or a csv file.
- The `--regex_string` option allows you to specify a regex to extract the subject ID from a file name. By default, this option is not active and results in splitting the file name by underscores, taking the first piece as the name.     If a regex string is provided, the first group match will be returned as the ID.

    Example:
    ```text
    filename -► G0003_set1_stand_abcdefg.jpg
    resulting subject ID -► G0003
    ```


- The `--matrix_file_type` option, if set, allows you to save the score matrix as a csv or npy file. If this option is not set, a score matrix will not be saved.
    
Sample run:
```
docker run --name low_og_aaf \
-v /home/xavier/Pictures/MORPH_Cloaking/Features/cloaked/low/aaf/templates/:/probe \
-v /home/xavier/Pictures/MORPH_Cloaking/Features/og/aaf/templates/:/gallery \
-v /home/xavier/Pictures/MORPH_Cloaking/Features/matching/low_og/aaf:/output \
featurematcher -p /probe -g /gallery -o /output \
--match_type all \
--group_name low_og_aaf
```