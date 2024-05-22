# from unifold.inference import *
import argparse
import os
import json
import torch
import time
from typing import *
import tqdm
from absl import logging
logging.set_verbosity("info")
from unifold.dataset import process
from ufconf.diffusion.diffuser import Diffuser
from ufconf.diffuse_ops import diffuse_inputs, make_noisy_quats, rbf_kernel
import copy
import random
import ufconf.utils as utils

max_retries = 3  # Maximum number of retries
retry_delay = 3  # Delay between retries in seconds

def process_replica(args, model, batch, diffuser, config, Job, output_traj_dir, job_name, replica):
    # set langevin step size
    sigma_l_r = 0.0005
    sigma_l_p = 0.01
    if "langevin_step" in Job:
        sigma_l_p = Job["langevin_step"]
        sigma_l_r = 0.05 * sigma_l_p
        
    s_0, f_0 = batch["aatype"].squeeze(), batch["true_frame_tensor"].squeeze()

    batch_constants = {
        key: batch[key].squeeze() for key in ("seq_mask", "residue_index", "chain_id")
    }
    rep_name = f"{job_name}"
    time_stamps = torch.linspace(Job["initial_t"], 0., Job["inf_steps"] + 1).float().to(args.device)
    utils.save_pdb(
        os.path.join(output_traj_dir, f"f0_{rep_name}.pdb"),
        s_0, f_0, **batch_constants, b_factor=None
    )
    iter_n = (tqdm.tqdm(range(Job["num_steps"]), total=Job["num_steps"]) \
        if Job["num_steps"] >= 10 else range(Job["num_steps"]))
    iter_t = (tqdm.tqdm(range(Job["inf_steps"]), total=Job["inf_steps"]) \
        if Job["inf_steps"] >= 10 else range(Job["inf_steps"]))
    s_t, f_t = batch["aatype"].squeeze(), batch["noisy_frames"].squeeze()
    
    for n_step in iter_n:
        # save noisy frames 
        if Job["save_trajectory"]:
            with open(os.path.join(output_traj_dir, f"ft_inf_{rep_name}_num{n_step}.pdb"), "a") as f:
                f.write(utils.to_pdb_string(
                    s_t, f_t, **batch_constants, model_id=0,
                ))
        with open(os.path.join(output_traj_dir, f"ft_inf_{rep_name}_rep{replica}.pdb"), "a") as f:
            f.write(utils.to_pdb_string(
                s_t, f_t, **batch_constants, model_id=n_step
            ))
        # denoise step at every langevin step
        batch_inner = copy.deepcopy(batch)
        for i in iter_t:
            t, s = time_stamps[i], time_stamps[i + 1]
            assert batch_inner["diffusion_t"] == t
            with torch.no_grad():
                out = model(batch_inner)
                if config.diffusion.chi.enabled:
                    pred_chi_angles_sin_cos = out["sm"]["angles"][-1,0,:,3:,:]
                ret_frames = out["pred_frame_tensor"]
            # aligne X0_h with X_t
            batch["noisy_frames"][0], out["pred_frame_tensor"] = utils.remove_center(batch["noisy_frames"][0], out["pred_frame_tensor"], mask=batch["frame_mask"])
            noisy_pos, noisy_mask = utils.compute_atomic_positions(batch["noisy_frames"], **batch_constants)
            noisy_pos = noisy_pos[0][0].reshape(-1,3)
            print("batch noisy_frames", batch["noisy_frames"].shape)
            print("pred frame shape", out["pred_frame_tensor"].shape)
            
            predicted_pos, predicted_mask = utils.compute_atomic_positions(out["pred_frame_tensor"], **batch_constants)
            predicted_pos = predicted_pos[0].reshape(-1,3)
            
            prediced_pos_aligned, U = utils.kabsch_rotate(predicted_pos, noisy_pos)
            prediced_pos_aligned = prediced_pos_aligned.reshape(1, batch["aatype"].shape[-1], -1, 3)
            batch["all_atom_positions"] = torch.tensor(prediced_pos_aligned,device = args.device)
            out["final_atom_positions"] = batch["all_atom_positions"] 
            processed_protein = utils.atom37_to_backb_frames(batch, 1e-8)
            ret_frames = processed_protein["backb_frames"]
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
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
                f_s_inner, tor_s_inner  = diffuser.denoise(
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
        
            if Job["save_trajectory"]:
                # output the predicted structures during reverse process into pdb files
                with open(os.path.join(output_traj_dir, f"f0h_inf_{rep_name}_num{n_step}.pdb"), "a") as f:
                    f.write(utils.to_pdb_string(
                        sh_0_inner, fh_0_innner, **batch_constants, model_id=i + 1,
                    ))
                # output the noisy structures during reverse process into pdb files
                with open(os.path.join(output_traj_dir, f"ft_inf_{rep_name}_num{n_step}.pdb"), "a") as f:
                    f.write(utils.to_pdb_string(
                        s_t_inner, f_t_inner, **batch_constants, model_id=i + 1,
                    ))
                    
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
        with torch.no_grad():
            out = model(batch_inner)
            ret_frames = out["pred_frame_tensor"]
        sh_0, fh_0 = batch_inner["aatype"].squeeze(), ret_frames.squeeze()
        if Job["save_trajectory"]:
            # output the predicted structures during reverse process into pdb files
            with open(os.path.join(output_traj_dir, f"f0h_inf_{rep_name}_num{n_step}.pdb"), "a") as f:
                f.write(utils.to_pdb_string(
                    sh_0, fh_0, **batch_constants, model_id=Job["inf_steps"] + 1,
                ))
        # output the predicted structure with only backbones
        with open(os.path.join(output_traj_dir, f"f0h_inf_{rep_name}_rep{replica}.pdb"), "a") as f:
            f.write(utils.to_pdb_string(
                    sh_0, fh_0, **batch_constants, model_id=n_step,
                ))
        f_s = diffuser.langevin(
            f_t, fh_0, batch["frame_gen_mask"],
            t=batch["diffusion_t"].squeeze(), sigma_l_r=torch.tensor(sigma_l_r, device=args.device), sigma_l_p=torch.tensor(sigma_l_p)
        )
        batch["noisy_frames"] = f_s
        s_t, f_t = batch["aatype"].squeeze(), batch["noisy_frames"].squeeze()
        # update features
        batch = make_noisy_quats(batch)

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
    
    diffuser = Diffuser(config.diffusion)
    gpu_diffuser = Diffuser(config.diffusion).to(args.device)
        
    config.data.predict.supervised = True
    config.globals.chunk_size = int(args.chunk_size)
    print("config chunk size", config.globals.chunk_size)
    
    job_name_list = list(job_configs.keys())
    Job_list = [job_configs[job_name] for job_name in job_name_list]

    # Iterate over each job
    for job_name, Job in zip(job_name_list, Job_list):
        output_dir, output_traj_dir, dir_feat_name = utils.setup_directories(args, job_name, Job, mode = "langevin")
        utils.save_config_json(checkpoint_path, output_dir, Job)
        # use existing PDB and MSA datasets to get all the features and labels for query Protein ID
        if not args.from_pdb:
            print(f"Please provide valid flag: 'from_pdb'.")
        # If use `from_pdb` flag, then generate MSA on the fly, get all the features and labels for query PDB file
        else:
            for attempt in range(max_retries):
                try:
                    # Attempt to load the features
                    feat_raw, all_chain_labels = utils.load_features_from_pdb(args, Job, dir_feat_name)
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
                    
        for replica in tqdm.tqdm(range(Job["num_replica"]), total=Job["num_replica"]) \
            if Job["num_replica"] >= 10 else range(Job["num_replica"]):
            logging.info(f"running replica {replica+1}/{Job['num_replica']}...")
            replica = replica + args.start * Job["num_replica"]
            my_seed = random.randint(0, 1000000)  # Generate a random seed
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
            featd, _ = utils.prepare_features(feat, Job)
            my_seed = random.randint(0, 1000000)  # Generate a random seed
            # diffuse inputs to noisy structure
            if featd["diffusion_t"] == 1.0:
                logging.info(f"set backbone and sidechain frame to be prior...")
                featd = utils.set_prior(featd, diffuser, my_seed, config.diffusion)
            else:
                logging.info(f"add noise {featd['diffusion_t']} to backbone and sidechain frame...")
                featd = diffuse_inputs(featd, diffuser, my_seed, config.diffusion, task="predict")
            batch = utils.prepare_batch(featd, lab, args)
            process_replica(args, model, batch, gpu_diffuser, config, Job, output_traj_dir, job_name, replica)
        print("Langevin inference completed!")
        
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path arguments
    parser.add_argument(
        "-t", "--tasks", type=str, default=None, required=True,
        help=r"""
    path to a json file specifying the task to conduct. The keyword arguments are illustrated in ufconf/README.md.
        """
    )
    parser.add_argument(
        "-i", "--input_pdbs", type=str, default=None,
        help="directory of input reference pdbs."
    )
    parser.add_argument(
        "-msa", "--msa_file_path", type=str, default=None,
        help="the path for existing MSA file"
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
        "--data_idx", type=str, default="0",
        help="specify random seed for MSA."
    )
    parser.add_argument(
        "--use_exist_msa", action="store_true",
        help="if set, then use existing MSA (default: not set)."
    )
    parser.add_argument(
        "--from_pdb", action=argparse.BooleanOptionalAction, default=True,
        help="if set, then running inference with MSA generated on the fly (default: set)."
    )
    parser.add_argument(
        "--bf16", action="store_true",
        help="if set, then bfloat16 is used. (default: not set)."
    )
    args = parser.parse_args()
    main(args)
