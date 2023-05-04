import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def list_files(path:str, allowed_ext:list) -> list:
    return [
        Path(os.path.join(dp, f)) 
        for dp, dn, filenames in os.walk(path) 
        for f in filenames 
        if os.path.splitext(f)[1] in allowed_ext
    ]


class Matcher:
    def __init__(self, probe_directory:str, gallery_directory:str=None, subject_id_regexp:str=None):
        p_templates = list_files(probe_directory, [".npy"])
        self.p_features = self.get_features(p_templates, "Probe Features: ")
        self.p_subjects = self.get_subject_ids(p_templates, subject_id_regexp, "Probe Subjects: ")
        self.p_labels = self.get_labels(p_templates, "Probe Labels: ")

        if gallery_directory is None:
            print(f"Matching {probe_directory} to {probe_directory}")
            g_templates = p_templates
            self.g_features = self.p_features
            self.g_subjects = self.p_subjects
            self.g_labels = self.p_labels
        else:
            print(f"Matching {probe_directory} to {gallery_directory}")
            g_templates = list_files(gallery_directory, [".npy"])
            self.g_features = self.get_features(g_templates, "Gallery Features: ")
            self.g_subjects = self.get_subject_ids(g_templates, subject_id_regexp, "Gallery Subjects: ")
            self.g_labels = self.get_labels(g_templates, "Gallery Labels: ")
            self.probe_equal_gallery = False


        num_probes, num_gallery = len(p_templates), len(g_templates)

        # This matrix has:
        #   0:  Impostors, 1:  Authentic, -1: Marked for Removal
        self.authentic_impostor = np.zeros(shape=(num_probes, num_gallery), dtype=np.int8)
        for i in range(num_probes):
            self.authentic_impostor[i, self.p_subjects[i] == self.g_subjects] = 1
            self.authentic_impostor[i, self.p_labels[i] == self.g_labels] = -1

            if gallery_directory is None:
                self.authentic_impostor[i, 0 : min(i + 1, num_gallery)] = -1
        
        self.matches = self.match_features(self.p_features, self.g_features)


    def get_labels(self, paths:list, description:str="Labels:") -> np.ndarray:
        return np.asarray([str(p.stem) for p in tqdm(paths, desc=description)])


    def get_features(self, feature_paths:list, description:str="Features:") -> np.ndarray:
        return np.asarray([ np.load(str(fp)) for fp in tqdm(feature_paths, desc=description) ])


    def get_subject_ids(self, feature_paths:list, regexp:str=None, description:str="Subjects:") -> np.array:
        def matcher(path, regexp=None):
            filename = str(Path(path).stem)
            if regexp is None:
                return filename.split("_")[0]
            else:
                match = re.search(regexp, filename)
                if not match: raise TypeError
                return match.group(0)
        
        return np.asarray([matcher(f, regexp) for f in tqdm(feature_paths, desc=description)])


    def match_features(self, probes:np.ndarray, gallery:np.ndarray) -> np.ndarray:
        return cosine_similarity(probes, gallery)


    def create_label_indices(self, labels) -> np.ndarray:
        indices = np.linspace(0, len(labels) - 1, len(labels)).astype(int)
        return np.transpose(np.vstack([indices, labels]))


    def get_indices_score(self, auth_or_imp):
        x, y = np.where(self.authentic_impostor == auth_or_imp)
        return np.transpose(
            np.vstack(
                [
                    x,
                    y,
                    np.round(self.matches[self.authentic_impostor == auth_or_imp], 6),
                ]
            )
        )


    def save_matches(self, output_directory:str, group_name:str, match_type:str="all", file_type="csv"):
        print("Saving matches output to " + f"{str(Path(output_directory))}")
        
        if match_type not in ["all", "authentic", "impostor"]: raise TypeError
        
        authentic_output = Path(output_directory) / f"{group_name}_authentic_scores.{file_type}"
        impostor_output = Path(output_directory) / f"{group_name}_impostor_scores.{file_type}"

        if match_type == "all":
            authentic, impostor = self.get_indices_score(1), self.get_indices_score(0)
            choices = [(authentic_output, authentic), (impostor_output, impostor)]
        elif match_type == "authentic":
            authentic = self.get_indices_score(1)
            choices = [(authentic_output, authentic)]
        elif match_type == "impostor":
            impostor = self.get_indices_score(0)
            choices = [(impostor_output, impostor)]

        for result_directory, data in choices:
            probe_labels = (self.p_labels[idx] for idx in np.int64(data[:,0]))
            gallery_labels = (self.g_labels[idx] for idx in np.int64(data[:,1]))
            scores = data[:,2]

            if file_type == "csv":
                delimiting_character = ","
            elif file_type == "txt":
                delimiting_character = " "
            else:
                raise NotImplementedError
            
            with open(result_directory, "w") as out:
                csv_out = csv.writer(out, delimiter=delimiting_character)
                csv_out.writerows(zip(probe_labels, gallery_labels, scores))
    

    def save_score_matrix(self, output_directory:str, group_name:str, file_type="csv"):
        print("Saving score matrix to " + f"{str(Path(output_directory))}")
        matrix_output = Path(output_directory) / f"{group_name}_score_matrix.{file_type}"

        if file_type == "csv":
            df = pd.DataFrame(self.matches, index=self.p_labels, columns=self.g_labels)
            df.to_csv(str(matrix_output))
        elif file_type == "npy":
            matrix_row_labels = str(Path(output_directory) / f"{group_name}_score_matrix_row_labels.txt")
            matrix_col_labels = str(Path(output_directory) / f"{group_name}_score_matrix_col_labels.txt")
            np.save(str(matrix_output), self.matches)
            np.savetxt(matrix_row_labels, self.p_labels, fmt="%s")
            np.savetxt(matrix_col_labels, self.g_labels, fmt="%s")


if __name__ == "__main__":
    description = """Feature Matcher for Biometric Templates
    
    The --probe_dir option provides a path to a directory containing templates in .npy format.
    The --gallery_dir option provides a path to a directory containing gallery templates in .npy format.
        If a path is not provided for gallery directory, this will assume the gallery equals the probe.
    
    The --output_dir option provides the path where the results (scores) are going to be saved.

    The --match_type option allows to perform matching and save scores for authentic, impostor, or both.
    The --group_name option allows you to set a name for the comparison or group.
    The --score_file_type option allows you to save the output in a text file or a csv file.
    The --regex_string option allows you to specify a regex to extract the subject ID from a file name.
        By default, this option is not active and results in splitting the file name by underscores, taking
        the first piece as the name.
        Example:
            filename -► G0003_set1_stand_abcdefg.jpg
            resulting subject ID -► G0003

        If a regex string is provided, the first group match will be returned as the ID.

    The --matrix_file_type option, if set, allows you to save the score matrix as a csv or npy file. 
        If this option is not set, a score matrix will not be saved.
"""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-p', '--probe_dir', type=str, help='path to the probe directory', required=True)
    parser.add_argument('-g', '--gallery_dir', type=str, help='path to the gallery directory')
    parser.add_argument('-o', '--output_dir', type=str, help='path to output directory', required=True)
    parser.add_argument('-m', '--match_type', type=str, choices=['authentic', 'impostor', 'all'], default='all', help='type of match to perform\n(default: all)')
    parser.add_argument('--matcher', type=str, choices=['cosinesimilarity'], default='cosinesimilarity', help='algorithm used to compare features\n(default: cosinesimilarity)')
    parser.add_argument('--score_file_type', type=str, choices=['csv', 'txt'], default='txt', help='type of scores file to output\n(default: txt)')
    parser.add_argument('--matrix_file_type', type=str, choices=['csv', 'npy'], default=None, help='type of score matrix file to output\n(default: None)')
    parser.add_argument('--group_name', type=str, default="group", help='name of the group or comparison\n(default: group)')
    parser.add_argument('-r', '--regex_string', type=str, default=None, help='regular expression to extract subject ID from files\n(default: None)')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True, parents=True) 

    m = Matcher(args.probe_dir, args.gallery_dir, args.regex_string)
    m.save_matches(args.output_dir, args.group_name, match_type=args.match_type, file_type=args.score_file_type)

    if args.matrix_file_type is not None:
        m.save_score_matrix(args.output_dir, args.group_name, args.matrix_file_type)
