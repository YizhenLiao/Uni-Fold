# from unifold.inference import *
import argparse
import os
import json
import torch
import time
from typing import *
import numpy as np
import tqdm

from absl import logging
logging.set_verbosity("info")
from ufconf.dataset import DiffoldDataset
from ufconf.diffusion.diffuser import Diffuser
from ufconf.diffuse_ops import diffuse_inputs, make_noisy_quats, rbf_kernel
from unifold.data.protein import to_pdb
import copy
import random
import ufconf.utils as utils
from unifold.dataset import process

max_retries = 3  # Maximum number of retries
retry_delay = 3  # Delay between retries in seconds

def setup_directories(args, job_name_list, job_name, Job):
    """
    Sets up and returns the paths for the main output directory, trajectory output directory, and features directory.

    Args:
        args: Parsed command-line arguments.
        job_name (str): Name of the current job.
        Job (dict): Configuration dictionary for the current job.

    Returns:
        tuple: A tuple containing paths to the output directory, trajectory directory, and features directory.
    """

    # Main output directory, named based on job details
    output_dir = os.path.join(args.outputs, "_".join(job_name_list), \
        f"tem{Job['initial_t']}_rep{Job['num_replica']}_num{Job['num_steps']}_inf{Job['inf_steps']}")
    os.makedirs(output_dir, exist_ok=True)

    # Directory for trajectory outputs
    output_traj_dir = os.path.join(output_dir, "traj")
    os.makedirs(output_traj_dir, exist_ok=True)

    # Directory for features
    dir_feat_name = os.path.join(args.outputs, "_".join(
            job_name_list), str(job_name) + '_features')
    os.makedirs(dir_feat_name, exist_ok=True)

    return output_dir, output_traj_dir, dir_feat_name

def save_config_json(args, checkpoint_path, output_dir, Job):
    """
    Saves the job configuration to a JSON file in the output directory.

    Args:
        args: Parsed command-line arguments.
        checkpoint_path (str): Path to the model checkpoint.
        output_dir (str): Directory where the output is to be saved.
        Job (dict): Configuration dictionary for the current job.
    """
    config_data = {
        "timestamp": time.strftime("%Y-%m-%d;%H:%M:%S", time.localtime()),
        "diffusion_t": Job["initial_t"],
        "checkpoint": checkpoint_path
    }

    # Define the file path
    config_file_path = os.path.join(output_dir, "config.json")

    # Writing the configuration data to the JSON file
    with open(config_file_path, 'w') as config_file:
        json.dump(config_data, config_file, indent=2)
        
def prepare_features(feat, Job):
    """
    Prepares features based on the job configuration.

    Args:
        feat (dict): Dictionary containing the initial feature data.
        Job (dict): Configuration dictionary for the current job.

    Returns:
        tuple: A tuple containing the modified feature dictionary and the residue list.
    """

    # Deep copy of the feature dictionary to avoid modifying the original
    featd = copy.deepcopy(feat)
    featd["diffusion_t"] = torch.tensor([Job["initial_t"]])
    featd["true_frame_tensor"] = featd["true_frame_tensor"].squeeze()

    # Handling residue indices if specified in Job configuration
    residue_list = None
    if "residue_idx" in Job:
        residue_list = []
        for part in Job["residue_idx"].split("/"):
            residue_idx = part.split("-")
            residue_indices = [int(i) for i in residue_idx]
            residue_list.append(residue_indices)

    # Generating mask for residues to be generated
    is_gen = torch.zeros_like(feat["frame_mask"])
    if residue_list is not None:
        for start, end in residue_list:
            is_gen[:, start:end + 1] = 1.
    else:
        is_gen = torch.ones_like(feat["frame_mask"])
    featd["frame_gen_mask"] = feat["frame_mask"] * is_gen

    return featd, residue_list

def prepare_batch(featd, lab, args):
    """
    Prepares the batch for processing by collating feature data and labels and transferring to the specified device.

    Args:
        featd (dict): Modified feature dictionary for a specific job and replica.
        lab (dict): Labels corresponding to the feature data.
        args: Parsed command-line arguments containing device information.

    Returns:
        dict: The batch data ready for processing, transferred to the specified device.
    """

    # Collate the features into a batch
    batch = DiffoldDataset.collater([featd])

    # Transfer batch data to the specified device (e.g., GPU)
    batch_device = {k: torch.as_tensor(v, device=args.device) for k, v in batch.items()}

    # Include additional batch data from labels
    chain_id_list = []
    for chain_index in range(len(lab)):
        chain_id = torch.tensor([chain_index + 1] * lab[chain_index]["aatype"].shape[0], device=args.device)
        chain_id_list.append(chain_id)
    batch_device["chain_id"] = torch.cat(chain_id_list, dim=0)

    return batch_device

def main(args):
    """
    Main function to run denoising inference for protein structures using ufconf model.

    This function takes a configuration file specifying various jobs, loads the ufconf model from a checkpoint,
    and processes each job according to the given configurations. 
    
    It supports:
    1. processing from a pre-existing PDB file and generating  MSA (Multiple Sequence Alignment) on the fly.
    2. using existing downloaded PDB dataset and MSA dataset to run inference.

    Parameters:
    - args: An argparse.Namespace object containing the following attributes:
        - tasks (str): Path to a JSON file specifying the tasks for protein structure inference.
        - input_pdbs (str): Directory of input reference PDB files.
        - outputs (str): Directory to save the output results.
        - checkpoint (str): Path to the model checkpoint file.
        - device (str): The device to run the model on (e.g., 'cuda:0').
        - chunk_size (str): Chunk size for processing.
        - model (str): Name of the ufconf model to use.
        - data_path (str): Path to the dataset containing PDB and MSA data.
        - data_idx (str): Index to specify the random seed for MSA.
        - from_pdb (bool): Flag to indicate whether to run inference with on-the-fly generated MSA.
        - bf16 (bool): Flag to indicate the use of bfloat16 precision.

    The function processes each job by loading features and labels, performing protein structure inference, and saving the results.
    It supports generating structures with different parameters such as diffusion time, number of replicas, and number of steps, 
    and saves trajectories and final structures in the specified output directory.
    """
    with open(args.tasks, "r") as f:
        job_configs = json.load(f)

    # load the models from checkpoint
    checkpoint_path = args.checkpoint
    tic = time.perf_counter()
    config, model = utils.config_and_model(args)
    toc = time.perf_counter() - tic
    logging.info(f"model initialized in {toc:.2f} seconds.")
    
    print("config beta_clip", config.diffusion.position.beta_clip)
    diffuser = Diffuser(config.diffusion)
    gpu_diffuser = Diffuser(config.diffusion).to(args.device)
        
    config.data.predict.supervised = True
    print("config global keys", config.globals.keys())
    config.globals.chunk_size = int(args.chunk_size)
    print("config chunk size", config.globals.chunk_size)
    
    job_name_list = list(job_configs.keys())
    Job_list = [job_configs[job_name] for job_name in job_name_list]

    feat_raw_list = []
    all_chain_labels_list = []
    # Iterate over each job
    for job_name, Job in zip(job_name_list, Job_list):
        output_dir, output_traj_dir, dir_feat_name = setup_directories(args, job_name_list, job_name, Job)

        save_config_json(args, checkpoint_path, output_dir, Job)
        
        # use existing PDB and MSA datasets to get all the features and labels for query Protein ID
        if not args.from_pdb:
            feat_raw, all_chain_labels = utils.load_features_from_lmdb(args, config, Job, job_name)
        # If use `from_pdb` flag, then generate MSA on the fly, get all the features and labels for query PDB file
        else:
            for attempt in range(max_retries):
                try:
                    # Attempt to load the features
                    feat_raw, all_chain_labels = utils.load_features_from_pdb(args, config, Job, dir_feat_name)
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
                
        feat_raw_list.append(feat_raw)
        all_chain_labels_list.append(all_chain_labels)
    
    for replica in tqdm.tqdm(range(Job["num_replica"]), total=Job["num_replica"]) \
        if Job["num_replica"] >= 10 else range(Job["num_replica"]):
        logging.info(f"running replica {replica+1}/{Job['num_replica']}...")
        # my_seed = random.randint(0, 1000000)  # Generate a random seed
        my_seed = 42  # Generate a fixed seed
        feat_list = []
        lab_list = []
        for index in range(len(feat_raw_list)):
            feat_raw = feat_raw_list[index]
            all_chain_labels = all_chain_labels_list[index]
            # preprocess all the MSA, make random deletion to the MSAs controlled by the random seed defined in `data_idx`
            feat, lab = process(
                config.data,
                mode="predict",
                features=feat_raw,
                labels=all_chain_labels,
                batch_idx=0,
                data_idx=my_seed,
                is_distillation=False
            )
            # print out the number of chains of the protein
            print("chain number", len(lab))
            featd, residue_list = prepare_features(feat, Job)
            print("featd frame gen mask", featd["frame_gen_mask"])
            
            feat_list.append(featd)
            lab_list.append(lab)
        
        # align the second stucture to the first one
        print("feat all atom pos shape", feat_list[0]["all_atom_positions"].shape)
        all_atom_pos_1 = feat_list[0]["all_atom_positions"][0].reshape(-1, 3)
        all_atom_pos_2 = feat_list[1]["all_atom_positions"][0].reshape(-1, 3)
        all_atom_pos_2_aligned, U = utils.kabsch_rotate(
            all_atom_pos_2, all_atom_pos_1)
        all_atom_pos_2_aligned = all_atom_pos_2_aligned.reshape(
            1, feat["aatype"].shape[-1], -1, 3)
        feat_list[1]["all_atom_positions"] = torch.tensor(all_atom_pos_2_aligned)
        processed_protein = utils.atom37_to_backb_frames(feat_list[1], 1e-8)
        feat_list[1]["true_frame_tensor"] = processed_protein["backb_frames"]
        replica = replica + args.start * Job["num_replica"]
        ft_list = []
        for job_name_index in range(len(job_name_list)):
            featd = feat_list[job_name_index]
            my_seed = random.randint(0, 1000000)  # Generate a random seed
            # my_seed = 10
            # diffuse inputs to noisy structure
            if featd["diffusion_t"] == 1.0:
                logging.info(f"set backbone and sidechain frame to be prior...")
                featd = utils.set_prior(featd, diffuser, my_seed, config.diffusion)
            else:
                logging.info(f"add noise {featd['diffusion_t']} to backbone and sidechain frame...")
                featd = diffuse_inputs(featd, diffuser, my_seed, config.diffusion, task="predict")

            batch = prepare_batch(featd, lab, args)
            print("batch chain id shape", batch["chain_id"].shape)
            
            job_name = job_name_list[job_name_index]
            rep_name = f"{job_name}"
            if replica == 0:
                s_0, f_0 = batch["aatype"].squeeze(), batch["true_frame_tensor"].squeeze()
                batch_constants = {
                    key: batch[key].squeeze() for key in ("seq_mask", "residue_index", "chain_id")
                }
                with open(os.path.join(output_traj_dir, f"f0_{rep_name}.pdb"), "a") as f:
                    f.write(utils.to_pdb_string(
                        s_0, f_0, **batch_constants, model_id=replica + 1
                    ))
                
            s_t, f_t = batch["aatype"].squeeze(), batch["noisy_frames"].squeeze()
            with open(os.path.join(output_traj_dir, f"ft_{rep_name}.pdb"), "a") as f:
                f.write(utils.to_pdb_string(
                    s_t, f_t, **batch_constants, model_id=replica + 1
                ))
            ft_list.append(f_t)
            
        iter_t = (tqdm.tqdm(range(Job["inf_steps"]), total=Job["inf_steps"]) \
        if Job["inf_steps"] >= 10 else range(Job["inf_steps"]))
        time_stamps = torch.linspace(Job["initial_t"], 0., Job["inf_steps"] + 1).float().to(args.device)
            
        print("ft_list 0 shape", ft_list[0].shape)
        print("ft_list 1 shape", ft_list[1].shape)
        interpolate_ft_list = utils.interpolate_conf(
            ft_list[0], ft_list[1], Job["num_steps"])
        print("interpolate ft list shape", len(interpolate_ft_list))
        print("interpolate ft list 0 shape", interpolate_ft_list[0].shape)
        rep_name = "_".join(job_name_list)
        if args.model == "ufconf_af2_v3_ftnx":
            with torch.no_grad():
                m_out, z_out, s_out = model.iteration_evoformer(batch)
        
        # inference from interpolated noisy frames
        for step in tqdm.tqdm(range(Job["num_steps"]), total=Job["num_steps"]) \
        if Job["num_steps"] >= 10 else range(Job["num_steps"]):
            batch["noisy_frames"] = interpolate_ft_list[step][None, None, ...]
            print("batch noisy frame shape", batch["noisy_frames"].shape)
            # update features
            batch = make_noisy_quats(batch)
            s_t, f_t = batch["aatype"].squeeze(
            ), batch["noisy_frames"].squeeze()
            # print("batch mask shape",batch["frame_gen_mask"].shape)
            # print("batch mask", batch["frame_mask"])
            # print("batch gen mask",batch["frame_gen_mask"])
            # print("batch diffusion t",batch["diffusion_t"])
            # print("batch time",batch["time_feat"])
            # output the noisy structures into pdb files
            with open(os.path.join(output_traj_dir, f"ft_inf_{rep_name}_rep{replica}.pdb"), "a") as f:
                f.write(utils.to_pdb_string(
                    s_t, f_t, **batch_constants, model_id=step + 1
                ))
            # denoise step at every langevin step
            batch_inner = copy.deepcopy(batch)
            for i in iter_t:
                # denoise step
                t, s = time_stamps[i], time_stamps[i + 1]
                print("batch inner diffusion t", batch_inner["diffusion_t"])
                assert batch_inner["diffusion_t"] == t
                if args.model == "ufconf_af2_v3_ftnx":
                    with torch.no_grad():
                        ret_atom_pos, ret_atom_mask, ret_frames = model.run_structure_module(batch, z_out, s_out)
                else:
                    with torch.no_grad():
                        out = model(batch_inner)
                        if config.diffusion.chi.enabled:
                            pred_chi_angles_sin_cos = out["sm"]["angles"][-1,0,:,3:,:]
                        ret_frames = out["pred_frame_tensor"]
                
                if config.diffusion.chi.enabled:
                    batch_inner["chi_angles_sin_cos"] = batch_inner["chi_angles_sin_cos"] *  batch_inner["chi_mask"][..., None]
                    pred_chi_angles_sin_cos = pred_chi_angles_sin_cos * batch_inner["chi_mask"][..., None]
                    tor_t_inner = batch_inner["noisy_chi_sin_cos"].squeeze()
                    tor_0_inner =  batch_inner["chi_angles_sin_cos"].squeeze()
                    torh_0_inner = pred_chi_angles_sin_cos.squeeze()
                else:
                    tor_t_inner = tor_0_inner = torh_0_inner = None
            
                sh_0_inner, fh_0_innner = batch_inner["aatype"].squeeze(), ret_frames.squeeze()
                f_t_inner = batch_inner["noisy_frames"].squeeze()
                
                with torch.no_grad():
                    if args.fwd:
                        f_s_inner, tor_s_inner  = gpu_diffuser.addnoise(
                            fh_0_innner,
                            batch_inner["frame_gen_mask"],
                            t=s,
                            tor_s=torh_0_inner
                        )
                    else:
                        f_s_inner, tor_s_inner  = gpu_diffuser.denoise(
                            f_t_inner, fh_0_innner,
                            batch_inner["frame_gen_mask"],
                            t=t, s=s,
                            tor_t=tor_t_inner, torh_0=torh_0_inner
                        )
                batch_inner["noisy_frames"] = f_s_inner
                if tor_s_inner is not None:
                    tor_s_inner = tor_s_inner * batch_inner["chi_mask"][...,None]
                    batch_inner["noisy_chi_sin_cos"] = tor_s_inner
                batch_inner["diffusion_t"] = torch.tensor([[s]], device=args.device)
                s_t_inner, f_t_inner = batch_inner["aatype"].squeeze(), batch_inner["noisy_frames"].squeeze()
                
                # update features
                batch_inner = make_noisy_quats(batch_inner)
                # print("batch diffusion t",batch["diffusion_t"])
                residue_t = batch_inner["diffusion_t"][..., None]
                # setting motif ts to 0
                residue_t = torch.where(
                    batch_inner["frame_gen_mask"] > 0., residue_t, torch.zeros_like(residue_t),
                )
                # setting unknown frame ts to 1
                residue_t = torch.where(
                    batch_inner["frame_mask"] > 0., residue_t, torch.ones_like(residue_t),
                )
                time_feat = rbf_kernel(residue_t, config.diffusion.d_time, 0., 1.)
                batch_inner["time_feat"] = time_feat
                
            # last step of the prediction
            t = time_stamps[Job["inf_steps"]]
            assert batch_inner["diffusion_t"] == t
            if args.model == "ufconf_af2_v3_ftnx":
                with torch.no_grad():
                    ret_atom_pos, ret_atom_mask, ret_frames = model.run_structure_module(batch, z_out, s_out)
            else:
                with torch.no_grad():
                    out = model(batch_inner)
                    ret_frames = out["pred_frame_tensor"]
            
            sh_0, fh_0 = batch_inner["aatype"].squeeze(), ret_frames.squeeze()
            b_factor = out["plddt"][..., None].tile(37).squeeze()
            
            # output the predicted interpolation structures into pdb files
            with open(os.path.join(output_traj_dir, f"f0h_inf_{rep_name}_rep{replica}.pdb"), "a") as f:
                f.write(utils.to_pdb_string(
                    sh_0, fh_0, **batch_constants, model_id=step + 1, b_factor=b_factor
                ))

        print("Complete interpolation process!")
        
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path arguments
    parser.add_argument(
        "-t", "--tasks", type=str, default=None, required=True,
        help=r"""
path to a json file specifying the task to conduct.
The json file should be organized as:
{
    "tid1": {"kw1": arg1, "kw2": arg2, ...},
    "tid2": {"kw1": arg1, "kw2", arg2, ...},
    ...
}.
The task id `tid` should contain only letters, digits, and `_`. 
The keyword arguments are illustrated in ufconf/README.md.
"""
    )
    parser.add_argument(
        "-i", "--input_pdbs", type=str, default=None,
        help="directory of input reference pdbs."
    )
    parser.add_argument(
        "-o", "--outputs", type=str, default=None, required=True,
        help="directory of outputs."
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, default=None,
        help="path to model checkpoint."
    )
    parser.add_argument(
        "-st", "--start", type=int, default=0,
        help="replica number start from"
    )

    # model arguments
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="the device to run the model. Please at any chance use GPUs. (default: `cuda:0`)."
    )
    parser.add_argument(
        "--chunk-size", type=str, default="1024",
        help="chunk size of the model"
    )
    parser.add_argument(
        "--model", type=str, default="ufconf_af2_v3_c",
        help="the name of the model. (default: `ufconf_af2_v3_c`)."
    )
    parser.add_argument(
        "--data_path", type=str, default="/mnt/vepfs/fs_projects/unifold/data_0916/traineval/",
        help="the path for the downloaded PDB and MSA dataset"
    )
    parser.add_argument(
        "--data_idx", type=str, default="0",
        help="specify random seed for MSA."
    )
    parser.add_argument(
        "--fwd", action="store_true",
        help="if set, then running reverse dynamics with forward process (default: not set)."
    )
    parser.add_argument(
        "--from_pdb", action="store_true",
        help="if set, then running inference with MSA generated on the fly (default: not set)."
    )
    parser.add_argument(
        "--bf16", action="store_true",
        help="if set, then bfloat16 is used. (default: not set)."
    )

    args = parser.parse_args()
    main(args)
