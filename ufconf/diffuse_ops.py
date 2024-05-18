from typing import *
from unicore.data.data_utils import numpy_seed

from unifold.modules.frame import rot_to_quat

import numpy as np
import torch

TensorDict = Dict[str, torch.Tensor]

def flip(prob):
    return int(np.random.rand() < prob)


def locally_continuous_mask(mask: torch.Tensor, keepprob: float, K: int):
    assert len(mask.shape) == 1
    beta_1 = 1. - 1. / K
    if keepprob < beta_1:
        beta_2 = keepprob / (1. - keepprob) / K
    else:
        beta_1 = beta_2 = keepprob

    p = keepprob
    ret = []
    for mc in mask.tolist():
        m = flip(p) * mc
        p = beta_1 if m else beta_2
        p = keepprob if mc == 0. else p    # reset p for hard 0s.
        ret.append(m)

    return torch.tensor(ret, dtype=mask.dtype)   # E(continuous 1s)=K


def make_generate_masks(
    frame_mask,
    task: str,
    K: int,
    obs_prob_max: float,
):
    if task in ("train", "eval_train"):
        obs_prob_kernel = lambda u: max(0., u - (1. - obs_prob_max))    # [0, pmax)
        obs_prob = obs_prob_kernel(np.random.rand())
        is_motif = locally_continuous_mask(frame_mask, obs_prob, K)
    else:
        is_motif = torch.zeros_like(frame_mask)
    frame_gen_mask = frame_mask * (1. - is_motif)
    return frame_gen_mask


default_ts = {
    "eval_init": 0.1,
    "eval_half": 0.5,
    "eval_last": 0.9,
}


def make_diffusion_t(
    task,
):
    if task in ("train", "eval_train"):
        t = np.random.rand()
    else:
        t = default_ts[task]
    return t


def rbf_kernel(
    r: torch.Tensor,    # [*]
    num_bins: int,
    r_min: float = 0,
    r_max: float = 1,
) -> torch.Tensor:      # [*, num_bins]
    r = r.clamp(max=r_max, min=r_min)   # clipping to range.
    bins = torch.linspace(
        r_min, r_max, num_bins,
        dtype=r.dtype, device=r.device
    )
    sigma = (r_max - r_min) / (num_bins - 1)
    rbf = torch.exp(-0.5 * ((r[..., None] - bins) / sigma) ** 2)
    return rbf


def make_noisy_quats(protein):
    assert "noisy_frames" in protein
    frames = protein["noisy_frames"]
    rot, trans = torch.split(frames[..., :3, :], (3, 1), dim=-1)
    quat = rot_to_quat(rot.cpu()).to(rot.device)
    protein["noisy_quats"] = torch.cat([quat, trans.squeeze(-1)], dim=-1)
    return protein



def diffuse_inputs(
    features, diffuser, seed, config, task
):
    frame_mask = features["frame_mask"]
    if task == "predict":
        t = features["diffusion_t"]
        frame_gen_mask = features["frame_gen_mask"]
    else:
        assert "diffusion_t" not in features
        assert "frame_gen_mask" not in features
        with numpy_seed(seed, 0, key="make_diffusion_parameters"):
            t = torch.tensor([make_diffusion_t(task)]) * config.max_diffusion_t
            frame_gen_mask = make_generate_masks(
                frame_mask.squeeze(0),
                task,
                config.motif_k,
                config.obs_prob_max
            )[None]
        features["diffusion_t"] = t
        features["frame_gen_mask"] = frame_gen_mask
    
    tor_s = features["chi_angles_sin_cos"] if config.chi.enabled else None

    with numpy_seed(seed, 0, key="diffuse_inputs"):
        noisy_frames, noisy_torsions = diffuser.addnoise(
            features["true_frame_tensor"],
            features["frame_gen_mask"],
            features["diffusion_t"],
            s=None,
            prior_mask=1. - frame_mask,
            tor_s=tor_s,
        )

    features["noisy_frames"] = noisy_frames
    if tor_s is not None:
        noisy_torsions = torch.nan_to_num(noisy_torsions, 0.)
        noisy_torsions = noisy_torsions * features["chi_mask"][..., None]
        features["noisy_chi_sin_cos"] = noisy_torsions
    features = make_noisy_quats(features)

    residue_t = t[..., None]
    # setting motif ts to 0
    residue_t = torch.where(
        frame_gen_mask > 0., residue_t, torch.zeros_like(residue_t),
    )
    # setting unknown frame ts to 1
    residue_t = torch.where(
        frame_mask > 0., residue_t, torch.ones_like(residue_t),
    )

    time_feat = rbf_kernel(residue_t, config.d_time, 0., 1.)
    features["time_feat"] = time_feat
    return features
