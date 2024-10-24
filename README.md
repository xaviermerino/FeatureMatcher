# Feature Matcher
Feature Matcher is a tool for matching biometric templates. It supports various options for handling probe and gallery directories, output formats, and matching types.

> Note: This version uses Dask. To use the previous, single-core, version, check out the `single-core` branch. 

## ⚙️ Options

Below is a detailed overview of the command-line options available in Feature Matcher:

| Option                | Short Option | Description                                                                                | Default  |
|-----------------------|--------------|--------------------------------------------------------------------------------------------|----------|
| `--probe-dir`         | `-p`         | Directory path containing .npy format templates.                                           | N/A |
| `--gallery-dir`       | `-g`         | Gallery directory path. Defaults to the probe directory if not specified.                  | N/A      |
| `--output-dir`        | `-o`         | Destination for results like scores and labels.                                            | N/A |
| `--match-type`        | `-m`         | Type of matching to perform (authentic, impostor, or all).                                 | `all`    |
| `--regex-subject`     | `-s`         | Regex pattern for extracting subject ID from file names.                                   | None     |
| `--regex-label`       | `-l`         | Regex pattern for extracting label ID from probe file names.                                     | None     |
| `--skip-labels`       | N/A          | Option to skip storing label pairs, useful for score-only scenarios.                       | `False`  |
| `--skip-matches`      | N/A          | Option to skip storing match scores, useful for label-only scenarios.                      | `False`  |
| `--persist`           | N/A          | Enables caching of intermediate results, recommended only with ample RAM.                  | `False`  |
| `--match-chunk-size`  | N/A          | Sets the number of probes processed per chunk.                                             | `64`     |
| `--mask-chunk-size`   | N/A          | Number of entries for mask generation per chunk.                                           | `256`    |
| `--auto-label-chunk-size`  | N/A     | Automatically determine the chunk size based on the number of comparisons.                     | False    |
| `--label-chunk-size`  | N/A          | Number of authentic and impostor label pairs processed per chunk                    | `8388608`    |
| `--authentic-label-chunk-size`  | N/A          | Number of authentic label pairs processed per chunk.                     | `8388608`    |
| `--impostor-label-chunk-size`  | N/A          | Number of impostor label pairs processed per chunk.                     | `1048576`    |

## 🐳 Getting Started on Docker

### ⚒️ Building the Docker Image

Clone the repository and navigate to its root.
```bash
git clone https://github.com/xaviermerino/FeatureMatcher.git
cd FeatureMatcher
```

Build the Docker image:

```bash
docker build -t xavier/featurematcher:latest .
docker tag xavier/featurematcher:latest featurematcher
```

### ⬇️ Downloading the Docker Image

Alternatively, download the docker image.
```bash
docker pull ghcr.io/xaviermerino/featurematcher:latest
docker tag ghcr.io/xaviermerino/featurematcher:latest featurematcher
```

### 🏃 Deploying the Feature Matcher

To deploy Feature Matcher in a Docker container for comparing probes against themselves and generating authentic, impostor scores, and labels, use the following command:

```bash
docker run --rm --name feature_matcher_instance \
--net host \
-v /path/to/probe/templates:/input \
-v /path/to/output:/output \
featurematcher \
--probe-dir /input \
--output-dir /output
```

This command:

- Runs the Feature Matcher in a Docker container named `feature_matcher_instance`.
- Mounts directories for probe templates, gallery templates, and output results.
- Opens up a dashboard to inspect progress on `https://127.0.0.1:8787/status`. The address might change if multiple instances are opened.
- Replace `/path/to/probe/templates` and `/path/to/output` with the actual paths to your probe templates and output directory, respectively.


### 🏃🏃 Advanced Usage: Handling Treated Gallery Images

For generating authentic and impostor scores, as well as label pairs, where gallery images have undergone treatments (like color correction), you can use a regex pattern to extract the correct label from the template filenames. This approach is particularly useful for distinguishing between treated and untreated images. Consider the following naming conventions for your templates:

- Probe templates: `046763_00F49_cc_cloaked_high.npy`
- Gallery templates: `046763_00F49_cc.npy`

With these naming conventions, the following command can be used:

```bash
docker run --name advanced_feature_matcher \
--net host \
-v /path/to/probe/templates:/input \
-v /path/to/gallery/templates:/gallery \
-v /path/to/output:/output \
featurematcher \
--probe-dir /input \
--gallery-dir /gallery \
--output-dir /output \
--regex-label '^(.*?)_cloaked_high$'
```

This command:

- Runs the Feature Matcher in a Docker container named `advanced_feature_matcher`.
- Mounts directories for probe templates, gallery templates, and output results.
- Opens up a dashboard to inspect progress on `https://127.0.0.1:8787/status`. The address might change if multiple instances are opened.
- Uses the `--regex-label` option with a regex pattern to correctly extract and match labels from probe filenames, distinguishing between color-corrected and original images.
- Replace `/path/to/probe/templates`, `/path/to/gallery/templates`, and `/path/to/output` with the actual paths to your probe templates, gallery templates, and output directory, respectively.

## 📈 Understanding the Output

The Feature Matcher in this version generates multiple components as output, all organized within a designated directory.

By default, the output directory structure appears as follows:
```
.
├── labels
│   ├── authentic
│   │   └── authentic_labels.csv
│   └── impostor
│       └── impostor_labels.csv
└── scores
    ├── authentic
    │   ├── 0.npy
    │   ├── 100.npy
    │   ├── 101.npy
    │   └── ...
    └── impostor
```

In this structure, the scores directory holds the different match types (i.e., authentic, impostor, or both), while the labels directory contains label pairs corresponding to these match types. 

To access the authentic scores, you can utilize the `dask` library to load the entire directory containing the `numpy` stack. If the dataset can fit into memory, you have the option to load it  all at once, producing a unified `numpy` array. An example of this process is provided below:

```python
import dask.array as da
import numpy as np

authentic_scores_da = da.from_npy_stack("/path/to/output/scores/authentic")
authentic_scores_np = np.asarray(authentic_scores_da)
```

Once the data is in this format, it becomes compatible with any scripts or applications that work with `numpy` arrays. If desired, you can save this data into a single file using the following code:

```python
np.save("authentic.npy", authentic_scores_np)
```
 

## 💨 Performance Optimization and Memory Management

Feature Matcher employs Dask to distribute computations across multiple workers in a single-machine cluster. Key to balancing performance and memory usage are the number of workers and the size of data chunks processed by each worker. You can fine-tune these settings using the `--match-chunk-size` and `--mask-chunk-size` options. For label generation, the options `--label-chunk-size`, `--authentic-label-chunk-size`, `--impostor-label-chunk-size`, and `--auto-label-chunk-size` can be used.


### 🔄 Default Configuration 
By default, the tool is configured to handle large datasets efficiently without excessive memory consumption. For example, with default settings, processing match scores for 155,000 probes against themselves is estimated to consume a maximum of 23 GiB of RAM. This computation uses 20 cores on an i7-12700K processor and takes about 8 minutes, assuming the results are saved to a fast PCIe SSD. Processing labels is also supported and is more time consuming since it is an I/O intensive process.

### 🔧 Adjusting Match Scores Chunk Sizes
To modify the performance characteristics, adjust the chunk size parameters. Larger chunks can improve computational speed but at the cost of increased memory usage. Conversely, smaller chunks reduce memory usage but may slow down the process.

Here's an example command incorporating modified chunk sizes for advanced use:

```bash
docker run --name advanced_feature_matcher \
--net host \
-v /path/to/probe/templates:/input \
-v /path/to/gallery/templates:/gallery \
-v /path/to/output:/output \
featurematcher \
--probe-dir /input \
--gallery-dir /gallery \
--regex-label '^(.*?)_cloaked_high$' \
--match-chunk-size 256 \
--mask-chunk-size 1024
```

In this example, `--match-chunk-size` is set to `256` and `--mask-chunk-size` to `1024`, which may be adjusted based on your machine's specifications and the dataset size.

### 🏷️ Adjusting Label Generation Chunk Sizes

Label generation is a memory intensive process. To tweak it to your system's specifications, you can make use of the `--label-chunk-size`, `--authentic-label-chunk-size`, `--impostor-label-chunk-size`, and `--auto-label-chunk-size` options. This section explains how to use each option.

#### Single Global Setting
You can provide a `--label-chunk-size` option with a number for both authentic and impostor pairs. This will make sure that each batch of labels is approximately that size. Smaller sizes will occupy less memory. See an example below:

```bash
docker run --rm --name feature_matcher_instance \
--net host \
-v /path/to/probe/templates:/input \
-v /path/to/output:/output \
featurematcher \
--probe-dir /input \
--output-dir /output \
--label-chunk-size 1000000
```

#### Different Authentic and Impostor Settings
If you would like to specify a size for authentic and impostor labels, you can provide the appropriate number with the `--authentic-label-chunk-size` and `--impostor-label-chunk-size` instead:
```bash
docker run --rm --name feature_matcher_instance \
--net host \
-v /path/to/probe/templates:/input \
-v /path/to/output:/output \
featurematcher \
--probe-dir /input \
--output-dir /output \
--authentic-label-chunk-size 1000000 \
--impostor-label-chunk-size 500000
```

#### Automatic Settings
Alternatively, you can let the program decide what would work best by estimating based on the number of comparisons that your dataset requires with the `--auto-label-chunk-size` option. This may not be optimal at times. 

```bash
docker run --rm --name feature_matcher_instance \
--net host \
-v /path/to/probe/templates:/input \
-v /path/to/output:/output \
featurematcher \
--probe-dir /input \
--output-dir /output \
--auto-label-chunk-size
```