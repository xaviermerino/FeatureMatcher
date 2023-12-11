# Feature Matcher
Feature Matcher is a tool for matching biometric templates. It supports various options for handling probe and gallery directories, output formats, and matching types.

## ⚙️ Options

Below is a detailed overview of the command-line options available in Feature Matcher:

| Option                | Short Option | Description                                                                                | Default  |
|-----------------------|--------------|--------------------------------------------------------------------------------------------|----------|
| `--probe-dir`         | `-p`         | Directory path containing .npy format templates.                                           | `/input` |
| `--gallery-dir`       | `-g`         | Gallery directory path. Defaults to the probe directory if not specified.                  | N/A      |
| `--output-dir`        | `-o`         | Destination for results like scores and labels.                                            | `/output`|
| `--match-type`        | `-m`         | Type of matching to perform (authentic, impostor, or all).                                 | `all`    |
| `--regex-subject`     | `-s`         | Regex pattern for extracting subject ID from file names.                                   | None     |
| `--regex-label`       | `-l`         | Regex pattern for extracting label ID from file names.                                     | None     |
| `--skip-labels`       | N/A          | Option to skip storing label pairs, useful for score-only scenarios.                       | `False`  |
| `--skip-matches`      | N/A          | Option to skip storing match scores, useful for label-only scenarios.                      | `False`  |
| `--persist`           | N/A          | Enables caching of intermediate results, recommended only with ample RAM.                  | `False`  |
| `--match-chunk-size`  | N/A          | Sets the number of probes processed per chunk.                                             | `64`     |
| `--mask-chunk-size`   | N/A          | Number of entries for mask generation per chunk.                                           | `256`    |
| `--label-chunk-size`  | N/A          | Number of label pairs processed per chunk (currently not implemented).                     | `512`    |

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
docker run --name feature_matcher_instance \
-v /path/to/probe/templates:/input \
-v /path/to/output:/output \
featurematcher
```

This command:

- Runs the Feature Matcher in a Docker container named `feature_matcher_instance`.
- Mounts directories for probe templates, gallery templates, and output results.
- Replace `/path/to/probe/templates` and `/path/to/output` with the actual paths to your probe templates and output directory, respectively.

### 🏃🏃 Advanced Usage: Handling Treated Gallery Images

For generating authentic and impostor scores, as well as label pairs, where gallery images have undergone treatments (like color correction), you can use a regex pattern to extract the correct label from the template filenames. This approach is particularly useful for distinguishing between treated and untreated images. Consider the following naming conventions for your templates:

- Probe templates: `046763_00F49_cc_cloaked_high.npy`
- Gallery templates: `046763_00F49_cc.npy`

With these naming conventions, the following command can be used:

```bash
docker run --name advanced_feature_matcher \
-v /path/to/probe/templates:/input \
-v /path/to/gallery/templates:/gallery \
-v /path/to/output:/output \
featurematcher \
--probe-dir /input \
--gallery-dir /gallery \
--regex-label '^(.*?)_cloaked_high$'
```

This command:

- Runs the Feature Matcher in a Docker container named `advanced_feature_matcher`.
- Mounts directories for probe templates, gallery templates, and output results.
- Uses the `--regex-label` option with a regex pattern to correctly extract and match labels from filenames, distinguishing between color-corrected and original images.
- Replace `/path/to/probe/templates`, `/path/to/gallery/templates`, and `/path/to/output` with the actual paths to your probe templates, gallery templates, and output directory, respectively.

## 💨 Performance Optimization and Memory Management

Feature Matcher employs Dask to distribute computations across multiple workers in a single-machine cluster. Key to balancing performance and memory usage are the number of workers and the size of data chunks processed by each worker. You can fine-tune these settings using the `--match-chunk-size` and `--mask-chunk-size` options. 

> *Note:* The `--label-chunk-size` option is currently not implemented, and its related code is commented out.

### 🔄 Default Configuration 
By default, the tool is configured to handle large datasets efficiently without excessive memory consumption. For example, with default settings, processing match scores for 155,000 probes against themselves is estimated to consume a maximum of 23 GiB of RAM. This computation uses 20 cores on an i7-12700K processor and takes about 8 minutes, assuming the results are saved to a fast PCIe SSD.

### 🔧 Adjusting Chunk Sizes
To modify the performance characteristics, adjust the chunk size parameters. Larger chunks can improve computational speed but at the cost of increased memory usage. Conversely, smaller chunks reduce memory usage but may slow down the process.

Here's an example command incorporating modified chunk sizes for advanced use:

```bash
docker run --name advanced_feature_matcher \
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
