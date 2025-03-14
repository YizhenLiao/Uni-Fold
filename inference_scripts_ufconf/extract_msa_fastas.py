# from unifold.inference import *
import argparse
import os
import json
import time
from typing import *
import tqdm
from absl import logging
import pickle

from typing import *

logging.set_verbosity("info")

import ufconf.utils as utils

max_retries = 3  # Maximum number of retries
retry_delay = 3  # Delay between retries in seconds

def main(args):
    with open(args.tasks, "r") as f:
        job_configs = json.load(f)
    
    job_name_list = list(job_configs.keys())
    Job_list = [job_configs[job_name] for job_name in job_name_list]

    # Iterate over each job
    for job_name, Job in zip(job_name_list, Job_list):
        output_dir, output_traj_dir, dir_feat_name = utils.setup_directories(args, job_name, Job, mode = "denoise")
        print("loading features from fasta...")
        for attempt in range(max_retries):
            try:
                # Attempt to load the features
                # seqs, feat_raw = utils.load_features_from_fasta(args, Job, dir_feat_name)
                # all_chain_labels = None
                feat_raw, all_chain_labels = utils.load_features_from_fasta(args, Job, dir_feat_name)
                # Save it as a .pkl file
                with open(os.path.join(dir_feat_name,'features.pkl'), 'wb') as f:
                    pickle.dump(feat_raw, f)
                break  # If successful, exit the loop
            except Exception as e:  # Catching a general exception
                if attempt < max_retries - 1:
                    # If not the last attempt, wait and then retry
                    print(f"EOFError encountered. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    # If it was the last attempt, re-raise the exception
                    print("Maximum retries reached. Aborting.")
                    raise
                
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path arguments
    parser.add_argument(
        "-t", "--tasks", type=str, default=None, required=True,
        help="path to a json file specifying the task to conduct"
    )
    parser.add_argument(
        "-i", "--input_pdbs", type=str, default=None,
        help="directory of input fasta files."
    )
    parser.add_argument(
        "-msa", "--msa_file_path", type=str, default=None,
        help="the path for existing MSA file"
    )
    parser.add_argument(
        "--use_exist_msa", action="store_true",
        help="if set, then use existing MSA (default: not set)."
    )
    parser.add_argument(
        "-o", "--outputs", type=str, default=None, required=True,
        help="directory of outputs."
    )

    args = parser.parse_args()
    main(args)
