import argparse
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union
from dask.diagnostics import ProgressBar
import dask
import dask.array as da
import dask.dataframe as dd
import dask.delayed as delayed
import numpy as np
from bokeh.util.warnings import BokehUserWarning
from dask.array.core import PerformanceWarning
from dask.distributed import Client
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import shutil

warnings.filterwarnings("ignore", category=BokehUserWarning)
warnings.filterwarnings(
    "ignore", message="Consider scattering data ahead of time and using futures."
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Sending large graph of size *"
)
warnings.filterwarnings(
    "ignore", category=PerformanceWarning, message="Sending large graph of size *"
)
warnings.filterwarnings(
    "ignore",
    category=PerformanceWarning,
    message="Increasing number of chunks by factor of *",
)
dask.config.set({"logging.distributed": "error"})


def list_files(path: str, allowed_ext: list) -> list:
    return [
        Path(os.path.join(dp, f))
        for dp, dn, filenames in os.walk(path)
        for f in filenames
        if os.path.splitext(f)[1] in allowed_ext
    ]


def get_labels(
    paths: list, regexp: str = None, description: str = "Labels:"
) -> np.ndarray:
    def matcher(path, regexp=regexp):
        filename = str(Path(path).stem)
        match = re.search(regexp, filename)
        if not match:
            raise TypeError
        return match.group(1)

    if regexp is None:
        return np.asarray([str(p.stem) for p in tqdm(paths, desc=description)])
    else:
        return np.asarray([matcher(f, regexp) for f in tqdm(paths, desc=description)])


def get_features(feature_paths: list, description: str = "Features:") -> np.ndarray:
    return np.asarray(
        [np.load(str(fp)) for fp in tqdm(feature_paths, desc=description)]
    )


def get_subject_ids(
    feature_paths: list, regexp: str = None, description: str = "Subjects:"
) -> np.ndarray:
    def matcher(path, regexp=None):
        filename = str(Path(path).stem)
        if regexp is None:
            return filename.split("_")[0]
        else:
            match = re.search(regexp, filename)
            if not match:
                raise TypeError
            return match.group(1)

    return np.asarray(
        [matcher(f, regexp) for f in tqdm(feature_paths, desc=description)]
    )


def daskified_cosine_similarity(X: da.Array, Y: np.ndarray, **kwargs: Any):
    if not isinstance(Y, np.ndarray):
        raise TypeError("`Y` must be a numpy array")

    chunks = (X.chunks[0], (len(Y),))
    return X.map_blocks(cosine_similarity, Y, dtype=np.float16, chunks=chunks, **kwargs)


def match_features(
    probes: np.ndarray, gallery: np.ndarray, match_chunk_size: int
) -> da.Array:
    probes = da.from_array(probes, chunks=({0: match_chunk_size, 1: -1}))
    similarity_matrix = daskified_cosine_similarity(probes, gallery)
    return similarity_matrix


def diagonal_mask(probe: da.Array, probe_index: da.Array, gallery: set) -> np.ndarray:
    probe = probe.ravel()
    probe_index = probe_index.ravel()
    num_probes = len(probe)
    gallery_size = len(gallery)

    result = np.empty((num_probes, gallery_size), dtype=np.bool_)

    for i in range(num_probes):
        p, index = probe[i], probe_index[i]
        if p not in gallery:
            result[i] = np.ones(gallery_size, dtype=np.bool_)
        else:
            result[i] = np.zeros(gallery_size, dtype=np.bool_)
            result[i, index:] = np.True_

    return result


def process_scores(
    match_type: str,
    matches: da.Array,
    mask: da.Array,
    output_directory: str,
):
    scores_directory = Path(output_directory) / "scores"
    scores_directory.mkdir(exist_ok=True, parents=True)
    score_file_path = str(scores_directory / match_type)

    scores = matches[mask.ravel()]
    print(f"Saving {match_type} scores to {score_file_path}")
    da.to_npy_stack(score_file_path, scores, axis=0)


# def process_labels(
#     match_type: str,
#     mask: da.Array,
#     p_labels_da: da.Array,
#     g_labels_da: da.Array,
#     output_directory: str,
#     label_chunk_size: int,
# ):
#     x, y = da.where(mask == True)

#     # This code calculates the chunks for later division
#     # using `label_chunk_size`. Commented out because
#     # performance is better without it at the moment.

#     # x.compute_chunk_sizes(), y.compute_chunk_sizes()
#     # x = x.rechunk(chunks=(label_chunk_size,))
#     # y = y.rechunk(chunks=(label_chunk_size,))

#     probe_label = p_labels_da[x]
#     gallery_label = g_labels_da[y]

#     label_pairs = da.concatenate(
#         [probe_label[:, None], gallery_label[:, None]],
#         axis=1,
#         allow_unknown_chunksizes=True,
#     )  # .rechunk(chunks=(label_chunk_size, 2))

#     labels_directory = Path(output_directory) / "labels"
#     labels_directory.mkdir(exist_ok=True, parents=True)
#     labels_file_path = str(labels_directory / match_type / f"{match_type}_labels_*.csv")

#     pairs_ddf = dd.from_dask_array(label_pairs, columns=["probe", "gallery"])
#     print(f"Saving {match_type} labels to {labels_file_path}")
#     pairs_ddf.to_csv(
#         labels_file_path,
#         index=False,
#         header=False,
#     )


def process_labels(
    match_type: str,
    mask: da.Array,
    p_labels_da: da.Array,
    g_labels_da: da.Array,
    output_directory: str,
    label_chunk_size: int = 128,
):
    import dask.dataframe as dd
    import pandas as pd

    x, y = da.where(mask == True)
    x.compute_chunk_sizes()
    y.compute_chunk_sizes()
    x = x.rechunk(chunks=(label_chunk_size,))
    y = y.rechunk(chunks=(label_chunk_size,))

    # Stack x and y into indices array
    indices = da.stack([x, y], axis=1).rechunk((label_chunk_size, 2))

    # Convert indices to Dask DataFrame
    df_indices = dd.from_dask_array(indices, columns=["x", "y"])

    labels_directory = Path(output_directory) / "labels" / match_type
    labels_directory.mkdir(parents=True, exist_ok=True)

    def process_partition(df_partition):
        x_chunk = df_partition["x"].values
        y_chunk = df_partition["y"].values

        # Extract labels for the chunk
        p_labels_chunk = da.take(p_labels_da, x_chunk, axis=0).compute()
        g_labels_chunk = da.take(g_labels_da, y_chunk, axis=0).compute()

        # Create a DataFrame with the label pairs
        label_pairs = pd.DataFrame(
            {"p_label": p_labels_chunk, "g_label": g_labels_chunk}
        )

        return label_pairs

    # Define the meta DataFrame to inform Dask about the output structure
    meta = pd.DataFrame(columns=["p_label", "g_label"])

    # Apply the function to each partition
    label_pairs_df = df_indices.map_partitions(process_partition, meta=meta)

    # Compute and collect the results
    label_pairs_df = label_pairs_df.compute()

    # Save the label pairs to the CSV file
    labels_file_path = labels_directory / f"{match_type}_labels.csv"
    label_pairs_df.to_csv(labels_file_path, index=False, header=False)

    print(f"Labels saved to {labels_file_path}")


# def process_chunk(
#     i: int,
#     indices_chunk: da.Array,
#     p_labels_da: da.Array,
#     g_labels_da: da.Array,
#     output_directory: Union[str, Path],
#     match_type: str,
# ) -> str:
#     """
#     Processes a single chunk of indices and writes the corresponding label pairs to a temporary file.

#     Parameters:
#     - i: The chunk index.
#     - indices_chunk: The chunk of indices (Dask array) to process.
#     - p_labels_da: The Dask array of predicted labels.
#     - g_labels_da: The Dask array of ground truth labels.
#     - output_directory: The directory where the output files will be saved.
#     - match_type: The type of match being processed, used for naming the output file.

#     Returns:
#     - str: The path to the temporary file where the label pairs are saved.
#     """
#     print(f"Process Chunk | Idx: {type(indices_chunk)} | P: {type(p_labels_da)} | G: {type(g_labels_da)}")
#     chunk_data = indices_chunk #.compute()  # Compute the chunk
#     x_chunk = chunk_data[:, 0]
#     y_chunk = chunk_data[:, 1]

#     # Extract labels for the chunk
#     p_labels_chunk = da.take(p_labels_da, x_chunk, axis=0)
#     g_labels_chunk = da.take(g_labels_da, y_chunk, axis=0)

#     # Stack labels into pairs
#     label_pairs = np.stack([p_labels_chunk, g_labels_chunk], axis=1)

#     # Create a temporary file path
#     print(f"Saving temporary file: temp_{match_type}_labels_chunk_{i}.csv")
#     temp_file_path = Path(output_directory) / f"temp_{match_type}_labels_chunk_{i}.csv"

#     # Save the label pairs to the temporary file
#     with open(temp_file_path, "w", newline="") as f:
#         np.savetxt(f, label_pairs, fmt="%s", delimiter=",")

#     return str(temp_file_path)


# def process_labels(
#     match_type: str,
#     mask: da.Array,
#     p_labels_da: da.Array,
#     g_labels_da: da.Array,
#     output_directory: str,
#     label_chunk_size: int = 128,
# ):
#     x, y = da.where(mask == True)
#     x.compute_chunk_sizes()
#     y.compute_chunk_sizes()
#     x = x.rechunk(chunks=(label_chunk_size,))
#     y = y.rechunk(chunks=(label_chunk_size,))

#     indices = da.stack([x, y], axis=1)
#     indices = indices.rechunk((label_chunk_size, 2))

#     labels_directory = Path(output_directory) / "labels" / match_type
#     labels_directory.mkdir(parents=True, exist_ok=True)

#     n_chunks = len(indices.chunks[0])

#     temp_files = []
#     for i in range(n_chunks):
#         indices_chunk = indices.blocks[i]
#         print(f"Process Labels | Idx: {type(indices_chunk)} | P: {type(p_labels_da)} | G: {type(g_labels_da)}")
#         temp_file = delayed(process_chunk)(
#             i, indices_chunk, p_labels_da, g_labels_da, labels_directory, match_type
#         )
#         temp_files.append(temp_file)

#     temp_file_paths = dask.compute(*temp_files)
#     labels_file_path = labels_directory / f"{match_type}_labels.csv"
#     print(f"Consolidating {len(temp_file_paths)} temporary files into {labels_file_path}...")

#     # Join all temporary files into the final CSV file
#     labels_file_path = labels_directory / f"{match_type}_labels.csv"
#     with open(labels_file_path, "w", newline="") as outfile:
#         for temp_file_path in tqdm(temp_file_paths, desc=f"Consolidating {match_type} labels"):
#             with open(temp_file_path, "r") as infile:
#                 shutil.copyfileobj(infile, outfile, length=10*1024*1024)  # Copy with a 10MB buffer
#             os.remove(temp_file_path)  # Remove the temporary file after it's merged

#     print(f"Labels saved to {labels_file_path}")


def load_data(
    directory: str,
    regexp_subject: str = None,
    regexp_label: str = None,
    skip_matches: bool = False,
    description="Probe",
) -> Tuple[List[Path], np.ndarray, np.ndarray, np.ndarray]:
    templates = sorted(list_files(directory, [".npy"]))
    features = None
    if not skip_matches:
        features = get_features(templates, f"{description} Features: ")
    subjects = get_subject_ids(
        templates, regexp=regexp_subject, description=f"{description} Subjects: "
    )
    labels = get_labels(templates, regexp=regexp_label, description="Labels: ")
    return templates, features, subjects, labels


def get_diagonal_mask(
    p_labels_da: da.Array,
    p_indices_da: da.Array,
    g_labels: set,
    probe_equal_gallery: bool,
    num_probes: int,
    num_gallery: int,
) -> da.Array:
    if probe_equal_gallery:
        return da.tri(
            num_probes,
            num_gallery,
            dtype=np.bool_,
            chunks=({0: "auto", 1: num_gallery}),
        )
    else:
        reshaped_labels = p_labels_da[:, None]
        reshaped_indices = p_indices_da[:, None]
        return reshaped_labels.map_blocks(
            diagonal_mask, reshaped_indices, g_labels, dtype=np.bool_
        )


def main(
    probe_directory: str,
    gallery_directory: str = None,
    output_directory: str = None,
    regexp_label: str = None,
    regexp_subject: str = None,
    match_type: str = "all",
    persist: bool = False,
    skip_labels: bool = False,
    skip_matches: bool = False,
    match_chunk_size: int = 64,
    mask_chunk_size: int = 256,
    label_chunk_size: int = 2048,
):
    if skip_labels and skip_matches:
        print("Nothing to do. Closing.")
        return

    # Create match and label plans
    match_plan = (
        ["match_authentic", "match_impostor"] if match_type == "all" else [match_type]
    )
    label_plan = (
        ["label_authentic", "label_impostor"] if match_type == "all" else [match_type]
    )
    match_plan = [] if skip_matches else match_plan
    label_plan = [] if skip_labels else label_plan
    plan = match_plan + label_plan

    p_templates, p_features, p_subjects, p_labels = load_data(
        directory=probe_directory,
        regexp_subject=regexp_subject,
        regexp_label=regexp_label,
        skip_matches=skip_matches,
    )
    probe_equal_gallery = (
        gallery_directory is None or probe_directory == gallery_directory
    )

    if probe_equal_gallery:
        g_templates, g_features, g_subjects, g_labels = (
            p_templates,
            p_features,
            p_subjects,
            p_labels,
        )
        print(f"Matching {probe_directory} to itself")
    else:
        g_templates, g_features, g_subjects, g_labels = load_data(
            directory=gallery_directory,
            regexp_subject=regexp_subject,
            regexp_label=None,
            skip_matches=skip_matches,
            description="Gallery",
        )
        print(f"Matching {probe_directory} to {gallery_directory}")

    with Client() as client:
        print("Dashboard: ", client.dashboard_link)
        try:
            # Convert to Dask arrays and chunk
            p_subjects_da = da.from_array(p_subjects, chunks=(mask_chunk_size,))
            g_subjects_da = da.from_array(g_subjects)
            p_labels_da = da.from_array(p_labels, chunks=(mask_chunk_size,))
            g_labels_da = da.from_array(g_labels)
            p_indices_da = da.arange(len(p_labels_da), chunks=(mask_chunk_size,))

            if not skip_matches:
                matches = match_features(
                    p_features, g_features, match_chunk_size
                ).ravel()

            # `mask_diagonal` marks redundant locations to avoid storing
            # reverse comparisons (e.g., A <=> B and B <=> A)
            mask_diagonal = get_diagonal_mask(
                p_labels_da=p_labels_da,
                p_indices_da=p_indices_da,
                g_labels=set(g_labels),
                probe_equal_gallery=probe_equal_gallery,
                num_probes=len(p_templates),
                num_gallery=len(g_templates),
            )

            # `mask_type` marks whether comparsion is mater or non-mated.
            # True denotes mated, False denotes non-mated.
            mask_type = p_subjects_da[:, None] == g_subjects_da

            # `mask_labels` marks whether comparison is invalid.
            # Invalid (True) means same image comparison.
            mask_labels = p_labels_da[:, None] == g_labels_da

            # `mask_authentic` is the final mask for mated comparisons.
            mask_authentic = da.bitwise_and(
                da.bitwise_xor(mask_type, mask_labels), mask_diagonal
            )

            # `mask_impostor` is the final mask for non-mated comparisons.
            mask_impostor = da.bitwise_and(da.bitwise_not(mask_type), mask_diagonal)

            if persist:
                if "match_authentic" in plan and "label_authentic" in plan:
                    mask_authentic.persist()
                if "match_impostor" in plan and "label_impostor" in plan:
                    mask_impostor.persist()
                if "match_authentic" in plan and "match_impostor" in plan:
                    matches.persist()

            # Processing Match Scores
            if match_plan:
                for mp in match_plan:
                    score_type = mp.split("_")[-1]
                    mask = (
                        mask_authentic if score_type == "authentic" else mask_impostor
                    )
                    process_scores(
                        match_type=score_type,
                        matches=matches,
                        mask=mask,
                        output_directory=output_directory,
                    )

                del matches

            # Processing Label Pairs
            if label_plan:
                p_labels = get_labels(
                    p_templates, regexp=None, description="Probe Labels: "
                )
                p_labels_da = da.from_array(p_labels, chunks=(label_chunk_size,))
                g_labels_da = g_labels_da.rechunk(chunks=(label_chunk_size,))

                print(f"Type: {type(p_labels_da)}")
                print(f"Type: {type(g_labels_da)}")
                for lp in label_plan:
                    label_type = lp.split("_")[-1]
                    mask = (
                        mask_authentic if label_type == "authentic" else mask_impostor
                    )
                    print(f"Processing {label_type} labels...")
                    process_labels(
                        match_type=label_type,
                        mask=mask,
                        output_directory=output_directory,
                        p_labels_da=p_labels_da,
                        g_labels_da=g_labels_da,
                        label_chunk_size=label_chunk_size,
                    )

                    del mask

            time.sleep(2)
            print("Done!")

        except Exception as e:
            print("[ERROR]", e)


if __name__ == "__main__":
    description = """Feature Matcher for Biometric Templates
    
    The --probe-dir option provides a path to a directory containing templates in .npy format.
    
    The --gallery-dir option provides a path to a directory containing gallery templates in .npy format.
        If a path is not provided for gallery directory, this will assume the gallery equals the probe.
    
    The --output-dir option provides the path where the results (scores, labels) are going to be saved.
    
    The --match-type option allows to perform matching and save scores for authentic, impostor, or both.
    
    The --regex-subject option allows you to specify a regex to extract the subject ID from a file name.
        By default, this option is not active and subjects are extracted by splitting the file name
        by underscores, taking the first piece as the name. Extensions are omitted.
        If a regex string is provided, the first group match will be returned as the ID.

        Example (no regex):
            filename -► G0003_set1_stand_abcdefg
            resulting subject ID -► G0003

        Example (with regex):
            filename -► G0003_set1_stand_abcdefg
            regex -► ^([^_]+)
            resulting subject ID -► G0003

    The --regex-label option allows you to specify a regex to extract the label ID from a file name.
        By default, this option is not active and the label is considered to be the file name. 
        This option is useful when a treatment has been applied to an image. Extensions are omitted.
        If a regex string is provided, the first group match will be returned as the ID.

        Example (no regex):
            filename -► G0003_set1_stand_abcdefg_color_corrected
            resulting label ID -► G0003_set1_stand_abcdefg_color_corrected

        Example (with regex):
            filename -► G0003_set1_stand_abcdefg_color_corrected
            regex -► ^(.*?)_color_corrected$
            resulting label ID -► G0003_set1_stand_abcdefg
    
    The --skip-labels option indicates not to store the label pairs. Label generation consumes
        significant memory and time.
    
    The --skip-matches option indicates not to store the match scores. May be useful when only
        label generation is required.

    The --persist option indicates to cache intermediate computation results to potentially speed up
        related computations. This is only advisable if you have enough memory, otherwise you may
        incur penalties for spilling intermediate results to disk.
        
    The --match-chunk-size option allows you to specify how many probes to process per chunk. 
        By default, this is set to 64. Feel free to increase/decrease it based on your setup.
        This option directly affects performance of match score calculation.
    
    The --mask-chunk-size option allows you to specify how many probes/gallery entries are 
        considered for mask (i.e., authentic, impostor) generation. By default this is set
        to 256. Feel free to increase/decrease it based on your setup.
        This option directly affects performance of authentic/impostor determination.

    The --label-chunk-size option allows you to specify how many label pairs entries are 
        generated at a time for export into CSV. By default this is set to 512. 
        Feel free to increase/decrease it based on your setup.
        This option directly affects performance of label pair generation.
        [Note: Not Implemented]
"""

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-p",
        "--probe-dir",
        type=str,
        help="path to the probe directory",
        required=True,
        default="/input",
    )
    parser.add_argument(
        "-g",
        "--gallery-dir",
        type=str,
        help="path to the gallery directory",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="path to output directory",
        required=True,
        default="/output",
    )
    parser.add_argument(
        "-m",
        "--match-type",
        type=str,
        choices=["authentic", "impostor", "all"],
        default="all",
        help="type of match to perform\n(default: all)",
    )
    parser.add_argument(
        "-s",
        "--regex-subject",
        type=str,
        default=None,
        help="regex to extract subject ID from files\n(default: None)",
    )
    parser.add_argument(
        "-l",
        "--regex-label",
        type=str,
        default=None,
        help="regex to extract label ID from files\n(default: None)",
    )
    parser.add_argument(
        "--match-chunk-size",
        type=int,
        help="chunk size for similarity matrix calculation\n(default: 64)",
        required=False,
        default=64,
    )
    parser.add_argument(
        "--mask-chunk-size",
        type=int,
        help="chunk size for calculating authentic and impostor masks\n(default: 256)",
        required=False,
        default=256,
    )
    parser.add_argument(
        "--label-chunk-size",
        type=int,
        help="chunk size for generating label pairs\n(default: 512) [NotImplemented]",
        required=False,
        default=2048,
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="if set, caches intermediate results for reuse.",
    )
    parser.add_argument(
        "--skip-labels", action="store_true", help="if set, skips saving labels."
    )
    parser.add_argument(
        "--skip-matches", action="store_true", help="if set, skips saving match scores."
    )

    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    main(
        probe_directory=args.probe_dir,
        gallery_directory=args.gallery_dir,
        output_directory=args.output_dir,
        regexp_label=args.regex_label,
        regexp_subject=args.regex_subject,
        match_type=args.match_type,
        persist=args.persist,
        skip_labels=args.skip_labels,
        skip_matches=args.skip_matches,
        match_chunk_size=args.match_chunk_size,
        mask_chunk_size=args.mask_chunk_size,
        label_chunk_size=args.label_chunk_size,
    )
