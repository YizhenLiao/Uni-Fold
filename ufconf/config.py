import ml_collections as mlc
import copy
from typing import Any

N_RES = "number of residues"
N_MSA = "number of MSA sequences"
N_EXTRA_MSA = "number of extra MSA sequences"
N_TPL = "number of templates"


d_pair = mlc.FieldReference(128, field_type=int)
d_msa = mlc.FieldReference(256, field_type=int)
d_template = mlc.FieldReference(64, field_type=int)
d_extra_msa = mlc.FieldReference(64, field_type=int)
d_single = mlc.FieldReference(384, field_type=int)
max_recycling_iters = mlc.FieldReference(0, field_type=int)
chunk_size = mlc.FieldReference(4, field_type=int)
aux_distogram_bins = mlc.FieldReference(64, field_type=int)
eps = mlc.FieldReference(1e-8, field_type=float)
inf = mlc.FieldReference(3e4, field_type=float)
use_templates = mlc.FieldReference(True, field_type=bool)
is_multimer = mlc.FieldReference(False, field_type=bool)


d_time = mlc.FieldReference(512, field_type=int)
scale_factor = mlc.FieldReference(10., field_type=float)
use_torsion_diffusion = mlc.FieldReference(False, field_type=bool)


def base_config():
    return mlc.ConfigDict(
        {
            "diffusion": {
                "position": {
                    "scale_factor": scale_factor,
                    "beta_clip": 0.99,
                    "kernel": "exp",
                    "params": (10., 2.5),
                    "eps": 1e-12,
                },
                "rotation": {
                    "max_sigma_sq": 9.,
                    "coef_a": 4.0,
                    "rw_approx_thres": 0.6,
                    "left": False,
                    "num_omega_bins": 1024,
                    "l_cutoff": 64,
                    "igso3_gaussian_thres": 0.2,
                    "igso3_grad_gaussian_thres": 0.6,
                },
                "chi": {
                    "enabled": use_torsion_diffusion,
                    "kernel": "exp",
                    "params": (16., 4.),
                },
                "motif_k": 15,
                "obs_prob_max": 0.6,
                "d_time": d_time,
                "max_diffusion_t": 1.0,
            },
            "data": {
                "common": {
                    "features": {
                        "aatype": [N_RES],
                        "all_atom_mask": [N_RES, None],
                        "all_atom_positions": [N_RES, None, None],
                        "alt_chi_angles": [N_RES, None],
                        "atom14_alt_gt_exists": [N_RES, None],
                        "atom14_alt_gt_positions": [N_RES, None, None],
                        "atom14_atom_exists": [N_RES, None],
                        "atom14_atom_is_ambiguous": [N_RES, None],
                        "atom14_gt_exists": [N_RES, None],
                        "atom14_gt_positions": [N_RES, None, None],
                        "atom37_atom_exists": [N_RES, None],
                        "frame_mask": [N_RES],
                        "true_frame_tensor": [N_RES, None, None],
                        "bert_mask": [N_MSA, N_RES],
                        "chi_angles_sin_cos": [N_RES, None, None],
                        "chi_mask": [N_RES, None],
                        "extra_msa_deletion_value": [N_EXTRA_MSA, N_RES],
                        "extra_msa_has_deletion": [N_EXTRA_MSA, N_RES],
                        "extra_msa": [N_EXTRA_MSA, N_RES],
                        "extra_msa_mask": [N_EXTRA_MSA, N_RES],
                        "extra_msa_row_mask": [N_EXTRA_MSA],
                        "is_distillation": [],
                        "msa_feat": [N_MSA, N_RES, None],
                        "msa_mask": [N_MSA, N_RES],
                        "msa_chains": [N_MSA, None],
                        "msa_row_mask": [N_MSA],
                        "num_recycling_iters": [],
                        "pseudo_beta": [N_RES, None],
                        "pseudo_beta_mask": [N_RES],
                        "residue_index": [N_RES],
                        "residx_atom14_to_atom37": [N_RES, None],
                        "residx_atom37_to_atom14": [N_RES, None],
                        "resolution": [],
                        "rigidgroups_alt_gt_frames": [N_RES, None, None, None],
                        "rigidgroups_group_exists": [N_RES, None],
                        "rigidgroups_group_is_ambiguous": [N_RES, None],
                        "rigidgroups_gt_exists": [N_RES, None],
                        "rigidgroups_gt_frames": [N_RES, None, None, None],
                        "seq_length": [],
                        "seq_mask": [N_RES],
                        "target_feat": [N_RES, None],
                        "template_aatype": [N_TPL, N_RES],
                        "template_all_atom_mask": [N_TPL, N_RES, None],
                        "template_all_atom_positions": [N_TPL, N_RES, None, None],
                        "template_alt_torsion_angles_sin_cos": [
                            N_TPL,
                            N_RES,
                            None,
                            None,
                        ],
                        "template_frame_mask": [N_TPL, N_RES],
                        "template_frame_tensor": [N_TPL, N_RES, None, None],
                        "template_mask": [N_TPL],
                        "template_pseudo_beta": [N_TPL, N_RES, None],
                        "template_pseudo_beta_mask": [N_TPL, N_RES],
                        "template_sum_probs": [N_TPL, None],
                        "template_torsion_angles_mask": [N_TPL, N_RES, None],
                        "template_torsion_angles_sin_cos": [N_TPL, N_RES, None, None],
                        "true_msa": [N_MSA, N_RES],
                        "use_clamped_fape": [],
                        "assembly_num_chains": [1],
                        "asym_id": [N_RES],
                        "sym_id": [N_RES],
                        "entity_id": [N_RES],
                        "num_sym": [N_RES],
                        "asym_len": [None],
                        "cluster_bias_mask": [N_MSA],
                    },
                    "masked_msa": {
                        "profile_prob": 0.1,
                        "same_prob": 0.1,
                        "uniform_prob": 0.1,
                    },
                    "block_delete_msa": {
                        "msa_fraction_per_block": 0.3,
                        "randomize_num_blocks": False,
                        "num_blocks": 5,
                        "min_num_msa": 16,
                    },
                    "random_delete_msa": {
                        "max_msa_entry": 1 << 25,  # := 33554432
                    },
                    "use_v2_weight": False,
                    "use_msa": True,
                    "use_self_as_extra_msa": False,
                    "v2_feature": False,
                    "gumbel_sample": False,
                    "max_extra_msa": 1024,
                    "msa_cluster_features": True,
                    "reduce_msa_clusters_by_max_templates": True,
                    "resample_msa_in_recycling": True,
                    "train_max_date": "2022-04-30",
                    "template_features": [
                        "template_all_atom_positions",
                        "template_sum_probs",
                        "template_aatype",
                        "template_all_atom_mask",
                    ],
                    "unsupervised_features": [
                        "aatype",
                        "residue_index",
                        "msa",
                        "msa_chains",
                        "num_alignments",
                        "seq_length",
                        "between_segment_residues",
                        "deletion_matrix",
                        "num_recycling_iters",
                        "crop_and_fix_size_seed",
                    ],
                    "recycling_features": [
                        "msa_chains",
                        "msa_mask",
                        "msa_row_mask",
                        "bert_mask",
                        "true_msa",
                        "msa_feat",
                        "extra_msa_deletion_value",
                        "extra_msa_has_deletion",
                        "extra_msa",
                        "extra_msa_mask",
                        "extra_msa_row_mask",
                        "is_distillation",
                    ],
                    "multimer_features": [
                        "assembly_num_chains",
                        "asym_id",
                        "sym_id",
                        "num_sym",
                        "entity_id",
                        "asym_len",
                        "cluster_bias_mask",
                    ],
                    "use_templates": use_templates,
                    "is_multimer": is_multimer,
                    "use_template_torsion_angles": use_templates,
                    "max_recycling_iters": max_recycling_iters,
                },
                "supervised": {
                    "use_clamped_fape_prob": 1.0,
                    "supervised_features": [
                        "all_atom_mask",
                        "all_atom_positions",
                        "resolution",
                        "use_clamped_fape",
                        "is_distillation",
                    ],
                },
                "predict": {
                    "fixed_size": True,
                    "subsample_templates": False,
                    "block_delete_msa": False,
                    "random_delete_msa": True,
                    "masked_msa_replace_fraction": 0.15,
                    "max_msa_clusters": 128,
                    "max_templates": 4,
                    "num_ensembles": 1,
                    "crop": False,
                    "crop_size": None,
                    "supervised": False,
                    "biased_msa_by_chain": False,
                    "share_mask": False,
                },
                "eval": {
                    "fixed_size": True,
                    "subsample_templates": False,
                    "block_delete_msa": False,
                    "random_delete_msa": True,
                    "masked_msa_replace_fraction": 0.15,
                    "max_msa_clusters": 128,
                    "max_templates": 4,
                    "num_ensembles": 1,
                    "crop": True,
                    "crop_size": 384,
                    "spatial_crop_prob": 0.5,
                    "ca_ca_threshold": 10.0,
                    "supervised": True,
                    "biased_msa_by_chain": False,
                    "share_mask": False,
                },
                "train": {
                    "fixed_size": True,
                    "subsample_templates": True,
                    "block_delete_msa": True,
                    "random_delete_msa": True,
                    "masked_msa_replace_fraction": 0.15,
                    "max_msa_clusters": 128,
                    "max_templates": 4,
                    "num_ensembles": 1,
                    "crop": True,
                    "crop_size": 256,
                    "spatial_crop_prob": 0.5,
                    "ca_ca_threshold": 10.0,
                    "supervised": True,
                    "use_clamped_fape_prob": 1.0,
                    "max_distillation_msa_clusters": 1000,
                    "biased_msa_by_chain": True,
                    "share_mask": True,
                },
            },
            "globals": {
                "chunk_size": chunk_size,
                "block_size": None,
                "d_pair": d_pair,
                "d_msa": d_msa,
                "d_template": d_template,
                "d_extra_msa": d_extra_msa,
                "d_single": d_single,
                "eps": eps,
                "inf": inf,
                "max_recycling_iters": max_recycling_iters,
                "alphafold_original_mode": False,
            },
            "model": {
                "is_multimer": is_multimer,
                "use_x_prev_first": True,
                "use_x_prev_last": False,
                "use_additional_recycler": False,
                "use_time_emb_per_layer": False,
                "input_embedder": {
                    "tf_dim": 22,
                    "msa_dim": 49,
                    "d_pair": d_pair,
                    "d_msa": d_msa,
                    "relpos_k": 32,
                    "max_relative_chain": 2,
                },
                "time_embedder": {
                    "d_in": d_time,
                    "d_msa": d_msa,
                    "d_pair": d_pair,
                    "init": "final",
                    "use_gated_linear": True,
                },
                "chi_embedder": {
                    "enabled": use_torsion_diffusion,
                    "d_in": 33, # 21+4*3
                    "d_out": d_msa,
                    "d_hid": d_msa,
                    "num_blocks": 4,
                    "init": "final",
                },
                "position_recycler": {
                    "d_pair": d_pair,
                    "d_hid": d_pair,
                    "num_blocks": 2,
                    "cutoff": 32.0,
                    "num_bins": 64,
                    "init": "final",        # zero init for best ft
                },
                "recycling_embedder": {
                    "d_pair": d_pair,
                    "d_msa": d_msa,
                    "min_bin": 3.25,
                    "max_bin": 20.75,
                    "num_bins": 15,
                    "inf": 1e8,
                },
                "template": {
                    "distogram": {
                        "min_bin": 3.25,
                        "max_bin": 50.75,
                        "num_bins": 39,
                    },
                    "template_angle_embedder": {
                        "d_in": 57,
                        "d_out": d_msa,
                    },
                    "template_pair_embedder": {
                        "d_in": 88,
                        "v2_d_in": [39, 1, 22, 22, 1, 1, 1, 1],
                        "d_pair": d_pair,
                        "d_out": d_template,
                        "v2_feature": False,
                    },
                    "template_pair_stack": {
                        "d_template": d_template,
                        "d_hid_tri_att": 16,
                        "d_hid_tri_mul": 64,
                        "num_blocks": 2,
                        "num_heads": 4,
                        "pair_transition_n": 2,
                        "dropout_rate": 0.25,
                        "inf": 1e9,
                        "tri_attn_first": True,
                    },
                    "template_pointwise_attention": {
                        "enabled": True,
                        "d_template": d_template,
                        "d_pair": d_pair,
                        "d_hid": 16,
                        "num_heads": 4,
                        "inf": 1e5,
                    },
                    "inf": 1e5,
                    "eps": 1e-6,
                    "enabled": use_templates,
                    "embed_angles": use_templates,
                },
                "extra_msa": {
                    "extra_msa_embedder": {
                        "d_in": 25,
                        "d_out": d_extra_msa,
                    },
                    "extra_msa_stack": {
                        "d_msa": d_extra_msa,
                        "d_pair": d_pair,
                        "d_hid_msa_att": 8,
                        "d_hid_opm": 32,
                        "d_hid_mul": 128,
                        "d_hid_pair_att": 32,
                        "num_heads_msa": 8,
                        "num_heads_pair": 4,
                        "num_blocks": 4,
                        "transition_n": 4,
                        "msa_dropout": 0.15,
                        "pair_dropout": 0.25,
                        "inf": 1e9,
                        "eps": 1e-10,
                        "outer_product_mean_first": False,
                        "no_col_attention": False,       # this is only enabled in no msa finetune.
                    },
                    "enabled": True,
                },
                "evoformer_stack": {
                    "d_msa": d_msa,
                    "d_pair": d_pair,
                    "d_hid_msa_att": 32,
                    "d_hid_opm": 32,
                    "d_hid_mul": 128,
                    "d_hid_pair_att": 32,
                    "d_single": d_single,
                    "num_heads_msa": 8,
                    "num_heads_pair": 4,
                    "num_blocks": 48,
                    "transition_n": 4,
                    "msa_dropout": 0.15,
                    "pair_dropout": 0.25,
                    "inf": 1e9,
                    "eps": 1e-10,
                    "outer_product_mean_first": False,
                    "no_col_attention": False,       # this is only enabled in no msa finetune.
                },
                "structure_module": {
                    "d_single": d_single,
                    "d_pair": d_pair,
                    "d_time": d_time,
                    "d_ipa": 16,
                    "d_angle": 128,
                    "num_heads_ipa": 12,
                    "num_qk_points": 4,
                    "num_v_points": 8,
                    "dropout_rate": 0.1,
                    "num_blocks": 8,
                    "no_transition_layers": 1,
                    "num_resnet_blocks": 2,
                    "num_angles": 7,
                    "trans_scale_factor": scale_factor,
                    "epsilon": 1e-12,
                    "inf": 1e5,
                    "separate_kv": False,
                    "ipa_bias": True,
                },
                "heads": {
                    "plddt": {
                        "num_bins": 50,
                        "d_in": d_single,
                        "d_hid": 128,
                        "enabled": True,
                    },
                    "distogram": {
                        "d_pair": d_pair,
                        "num_bins": aux_distogram_bins,
                        "disable_enhance_head": False,
                        "enabled": True,
                    },
                    "pae": {
                        "d_pair": d_pair,
                        "num_bins": aux_distogram_bins,
                        "enabled": False,
                        "iptm_weight": 0.8,
                        "disable_enhance_head": False,
                    },
                    "masked_msa": {
                        "d_msa": d_msa,
                        "d_out": 23,
                        "disable_enhance_head": False,
                        "enabled": True,
                    },
                    "experimentally_resolved": {
                        "d_single": d_single,
                        "d_out": 37,
                        "enabled": False,
                        "disable_enhance_head": False,
                    },
                },
            },
            "loss": {
                "time_scaling_method": "exp",
                "time_scaling_coef_a": 1.0,
                "distogram": {
                    "min_bin": 2.3125,
                    "max_bin": 21.6875,
                    "num_bins": 64,
                    "eps": 1e-6,
                    "weight": 0.3,
                    "time_scaling": 1.0,
                },
                "experimentally_resolved": {
                    "eps": 1e-8,
                    "min_resolution": 0.1,
                    "max_resolution": 3.0,
                    "weight": 0.0,
                },
                "fape": {
                    "backbone": {
                        "clamp_distance": 10.0,
                        "clamp_distance_between_chains": 30.0,
                        "loss_unit_distance": 10.0,
                        "loss_unit_distance_between_chains": 20.0,
                        "weight": 0.5,
                        "eps": 1e-4,
                    },
                    "sidechain": {
                        "clamp_distance": 10.0,
                        "length_scale": 10.0,
                        "weight": 0.5,
                        "eps": 1e-4,
                    },
                    "weight": 1.0,
                    "time_scaling": 1.0,
                },
                "plddt": {
                    "min_resolution": 0.1,
                    "max_resolution": 3.0,
                    "cutoff": 15.0,
                    "num_bins": 50,
                    "eps": 1e-10,
                    "weight": 0.01,
                    "time_scaling": 1.0,
                },
                "masked_msa": {
                    "eps": 1e-8,
                    "weight": 2.0,
                },
                "supervised_chi": {
                    "chi_weight": 0.5,
                    "angle_norm_weight": 0.01,
                    "eps": 1e-6,
                    "weight": 1.0,
                    "time_scaling": 1.0,
                },
                "violation": {
                    "violation_tolerance_factor": 12.0,
                    "clash_overlap_tolerance": 1.5,
                    "bond_angle_loss_weight": 0.3,
                    "eps": 1e-6,
                    "weight": 0.0,
                    "time_scaling": 1.0,
                },
                "pae": {
                    "max_bin": 31,
                    "num_bins": 64,
                    "min_resolution": 0.1,
                    "max_resolution": 3.0,
                    "eps": 1e-8,
                    "weight": 0.0,
                    "time_scaling": 1.0,
                },
                "repr_norm": {
                    "weight": 0.01,
                    "tolerance": 1.0,
                },
                "chain_centre_mass": {
                    "weight": 0.0,
                    "eps": 1e-8,
                },
                "update_penalty": {
                    "weight": 0.0,
                    "eps": 1e-8,
                    "time_scaling": 1.0,
                },
                "global_atom_error": {
                    "clamp_distance": 10.0,
                    "length_scale": 10.0,
                    "weight": 0.0,
                    "eps": 1e-4,
                    "time_scaling": 1.0,
                }
            },
        }
    )


def recursive_set(c: mlc.ConfigDict, key: str, value: Any, ignore: str = None):
    with c.unlocked():
        for k, v in c.items():
            if ignore is not None and k == ignore:
                continue
            if isinstance(v, mlc.ConfigDict):
                recursive_set(v, key, value)
            elif k == key:
                c[k] = value


def model_config(name, train=False):
    c = copy.deepcopy(base_config())

    if name == "ufconf_af2_v3":
        c = ufconf_af2_v3(c)
    elif name == "ufconf_af2_v3_ft":
        c = ufconf_af2_v3(c)
        c.loss.violation.weight = 0.5
        c.data.train.crop_size = 384
    elif name == "ufconf_af2_v3_ftnx":
        c = ufconf_af2_v3(c)
        c.loss.violation.weight = 0.5
        c.data.train.crop_size = 384
        c.model.use_x_prev_first = False
    elif name == "ufconf_af2_v3_ftlx":
        c = ufconf_af2_v3(c)
        c.loss.violation.weight = 0.5
        c.data.train.crop_size = 384
        c.model.use_x_prev_first = False
        c.model.use_x_prev_last = True
    elif name == "ufconf_af2_v3_b":
        c = ufconf_af2_v3_b(c)
    elif name == "ufconf_af2_v3_b2":
        c = ufconf_af2_v3_b2(c)
    elif name == "ufconf_af2_v3_b2_nomsa":
        c = ufconf_af2_v3_b2(c)
        c = nomsa(c)
    elif name == "ufconf_af2_v3_b3_fast":
        c = ufconf_af2_v3_b3(c)
        c = fast(c)
    elif name == "ufconf_af2_v3_b3_veryfast":
        c = ufconf_af2_v3_b3(c)
        c = veryfast(c)
    elif name == "ufconf_af2_v3_b3_veryfast_nomask":
        c = ufconf_af2_v3_b3(c)
        c = veryfast(c)
        c = close_masked_msa(c)
    elif name == "ufconf_af2_v3_b4_fjh":
        c = ufconf_af2_v3_b4_fjh(c)
    elif name == "ufconf_af2_v3_b5_fjh":
        c = ufconf_af2_v3_b5_fjh(c)
    elif name == "ufconf_af2_v3_b6_fjh":
        c = ufconf_af2_v3_b6_fjh(c)
    elif name == "ufconf_af2_v3_b7_fjh":
        c = ufconf_af2_v3_b7_fjh(c)
    elif name == "ufconf_af2_v3_b8_fjh":
        c = ufconf_af2_v3_b8_fjh(c)
    elif name == "ufconf_af2_v3_b1p_nomsav2":
        c = ufconf_af2_v3_b1p(c)
        c = nomsa_v2(c)
    elif name == "ufconf_af2_v3_b1pf_nomsav2":
        c = ufconf_af2_v3_b1p(c)
        c = nomsa_v2(c)
        c.loss.time_scaling_coef_a = 3.5
    elif name == "ufconf_af2_v3_b1pf":
        c = ufconf_af2_v3_b1p(c)
        c.loss.time_scaling_coef_a = 3.5
    elif name == "ufconf_af2_v3_b3_nomsav2":
        c = ufconf_af2_v3_b3(c)
        c = nomsa_v2(c)
    elif name == "ufconf_af2_v3_b4_nomsav2":
        c = ufconf_af2_v3_b4(c)
        c = nomsa_v2(c)
    elif name == "ufconf_af2_v3_b4_veryfast":
        c = ufconf_af2_v3_b4(c)
        c = veryfast(c)
    elif name == "ufconf_af2_v3_c":
        c = ufconf_af2_v3_c(c)
    elif name == "ufconf_af2_v3_c_256":
        c = ufconf_af2_v3_c(c)
        recursive_set(c, "max_msa_clusters", 256)
        recursive_set(c, "max_extra_msa", 1024)
    elif name == "ufconf_af2_v3_c_128":
        c = ufconf_af2_v3_c(c)
        recursive_set(c, "max_msa_clusters", 128)
        recursive_set(c, "max_extra_msa", 512)
    elif name == "ufconf_af2_v3_c_64":
        c = ufconf_af2_v3_c(c)
        recursive_set(c, "max_msa_clusters", 64)
        recursive_set(c, "max_extra_msa", 256)
    elif name == "ufconf_af2_v3_c_32":
        c = ufconf_af2_v3_c(c)
        recursive_set(c, "max_msa_clusters", 32)
        recursive_set(c, "max_extra_msa", 128)
    elif name == "ufconf_af2_v3_c_fast":
        c = ufconf_af2_v3_c(c)
        c = fast(c)
    elif name == "ufconf_af2_v3_c_tor":
        c = ufconf_af2_v3_c(c)
        c = tordiff(c)
    elif name == "ufconf_af2_v3_c_tor_64":
        c = ufconf_af2_v3_c(c)
        c = tordiff(c)
        recursive_set(c, "max_msa_clusters", 64)
        recursive_set(c, "max_extra_msa", 256)
    elif name == "ufconf_af2_v3_c_v2w":
        c = ufconf_af2_v3_c(c)
        c.data.common.use_v2_weight = True
    elif name == "ufconf_af2_v3_c_tor_v2w":
        c = ufconf_af2_v3_c(c)
        c = tordiff(c)
        c.data.common.use_v2_weight = True
    elif name == "ufconf_af2_v3_c_tor_v2w_256":
        c = ufconf_af2_v3_c(c)
        c = tordiff(c)
        c.data.common.use_v2_weight = True
        recursive_set(c, "max_msa_clusters", 256)
        recursive_set(c, "max_extra_msa", 1024)
    elif name == "ufconf_af2_v3_c_tor_v2w_128":
        c = ufconf_af2_v3_c(c)
        c = tordiff(c)
        c.data.common.use_v2_weight = True
        recursive_set(c, "max_msa_clusters", 128)
        recursive_set(c, "max_extra_msa", 512)
    elif name == "ufconf_af2_v3_c_tor_v2w_64":
        c = ufconf_af2_v3_c(c)
        c = tordiff(c)
        c.data.common.use_v2_weight = True
        recursive_set(c, "max_msa_clusters", 64)
        recursive_set(c, "max_extra_msa", 256)
    elif name == "ufconf_af2_v3_c_tor_v2w_32":
        c = ufconf_af2_v3_c(c)
        c = tordiff(c)
        c.data.common.use_v2_weight = True
        recursive_set(c, "max_msa_clusters", 32)
        recursive_set(c, "max_extra_msa", 128)
    else:
        raise ValueError(f"invalid --model-name: {name}.")

    if train:
        c.globals.chunk_size = None

    recursive_set(c, "inf", 3e4)
    recursive_set(c, "eps", 1e-5, "loss")
    return c


def ufconf_af2_v3(c):
    recursive_set(c, "max_extra_msa", 2048)
    recursive_set(c, "max_msa_clusters", 512)
    recursive_set(c, "is_multimer", True)
    recursive_set(c, "v2_feature", True)
    recursive_set(c, "gumbel_sample", True)
    recursive_set(c, "use_templates", False)
    c.model.template.template_angle_embedder.d_in = 34
    c.model.template.template_pair_stack.tri_attn_first = False
    c.model.template.template_pointwise_attention.enabled = False
    c.model.heads.pae.enabled = True
    c.model.heads.experimentally_resolved.enabled = True
    c.model.heads.masked_msa.d_out = 22
    c.model.structure_module.separate_kv = True
    c.model.structure_module.ipa_bias = False
    c.model.structure_module.trans_scale_factor = 20.
    c.loss.pae.weight = 0.1
    c.loss.violation.weight = 0.02
    c.loss.experimentally_resolved.weight = 0.01
    c.model.input_embedder.tf_dim = 21
    c.globals.alphafold_original_mode = True
    c.data.train.crop_size = 256
    c.loss.repr_norm.weight = 0.
    c.loss.chain_centre_mass.weight = 0.0
    c.loss.update_penalty.weight = 0.01
    c.loss.global_atom_error.weight = 0.5
    recursive_set(c, "outer_product_mean_first", True)
    c.data.common.reduce_msa_clusters_by_max_templates = False  # no templs
    return c

def ufconf_af2_v3_b(c):
    c = ufconf_af2_v3(c)
    # new diffusion curation (check notebooks/test_diffusion_scheme.ipynb)
    c.diffusion.position.kernel = "ddpm"
    c.diffusion.position.params = (0.01, 1.5)
    # lower global atom error weight (as a guidance loss to learn optimal aligment, while keeping main acc ctrl.ed by fape.)
    c.loss.global_atom_error.weight = 0.1
    # add time scaling (scale down loss at t->1. Therefore ~10x learning rate (1e-3) can be applied.)
    recursive_set(c, "time_scaling", 0.1)
    # use spatial crop only
    recursive_set(c, "spatial_crop_prob", 1.0)
    # other changes:
    # 1. remove self distillation training.
    # 2. use UnifoldDataset to correctly sampling monomers.
    # 3. implemented multimer dataset correctly (ctrl.ed via finetune_ufconf.sh)
    return c

def ufconf_af2_v3_b1p(c):
    c = ufconf_af2_v3(c)
    # new diffusion curation (check notebooks/test_diffusion_scheme.ipynb)
    c.diffusion.position.kernel = "ddpm"
    c.diffusion.position.params = (0.01, 1.5)
    # lower global atom error weight (as a guidance loss to learn optimal aligment, while keeping main acc ctrl.ed by fape.)
    c.loss.global_atom_error.weight = 0.1
    # add time scaling (scale down loss at t->1. Therefore ~10x learning rate (1e-3) can be applied.)
    recursive_set(c, "time_scaling", 0.01)
    c.loss.time_scaling_coef_a = 0.5
    # use spatial crop only
    recursive_set(c, "spatial_crop_prob", 1.0)
    c.loss.update_penalty.weight = 0.1
    # other changes:
    # 1. remove self distillation training.
    # 2. use UnifoldDataset to correctly sampling monomers.
    # 3. implemented multimer dataset correctly (ctrl.ed via finetune_ufconf.sh)
    return c

def ufconf_af2_v3_b2(c):
    c = ufconf_af2_v3_b(c)
    # slower noise growth
    c.diffusion.position.params = (0.01, 2.0)
    # encourage variance
    c.loss.update_penalty.weight = 0.1
    # add time scaling (scale down loss at t->1. Therefore ~10x learning rate (1e-3) can be applied.)
    recursive_set(c, "time_scaling", 0.2)
    c.loss.global_atom_error.time_scaling = 0.01
    # use spatial crop only
    recursive_set(c, "spatial_crop_prob", 1.0)
    c.loss.time_scaling_coef_a = 0.5
    return c

def ufconf_af2_v3_b3(c):
    # truncate diffusion to [0, 0.6]
    c = ufconf_af2_v3_b2(c)
    c.diffusion.max_diffusion_t = 0.6
    return c

def ufconf_af2_v3_b4_fjh(c):
    c = ufconf_af2_v3_b(c)
    # add time scaling (scale down loss at t->1. Therefore ~10x learning rate (1e-3) can be applied.)
    recursive_set(c, "time_scaling", 0.01)
    return c

def ufconf_af2_v3_b5_fjh(c):
    c = ufconf_af2_v3_b(c)
    # slower noise growth
    c.diffusion.position.coef_a = 2.0
    # encourage variance
    c.loss.update_penalty.weight = 0.1
    return c

def ufconf_af2_v3_b6_fjh(c):
    c = ufconf_af2_v3_b(c)
    # add time scaling (scale down loss at t->1. Therefore ~10x learning rate (1e-3) can be applied.)
    recursive_set(c, "time_scaling", 0.05)
    return c

def ufconf_af2_v3_b7_fjh(c):
    c = ufconf_af2_v3_b(c)
    # slower noise growth
    c.diffusion.position.coef_a = 2.0
    return c

def ufconf_af2_v3_b8_fjh(c):
    c = ufconf_af2_v3_b(c)
    # add time scaling (scale down loss at t->1. Therefore ~10x learning rate (1e-3) can be applied.)
    recursive_set(c, "time_scaling", 0.02)
def ufconf_af2_v3_b4(c):
    c = ufconf_af2_v3(c)
    c.diffusion.position.kernel = "linear"
    c.diffusion.position.params = (2.0, 0.2, 0.9, 0.7)
    c.loss.global_atom_error.weight = 0.1
    recursive_set(c, "spatial_crop_prob", 1.0)
    c.loss.update_penalty.weight = 0.1
    # add time scaling (scale down loss at t->1. Therefore ~10x learning rate (1e-3) can be applied.)
    recursive_set(c, "time_scaling", 0.2)
    c.loss.global_atom_error.time_scaling = 0.01
    # use spatial crop only
    c.loss.time_scaling_coef_a = 0.5
    return c

def nomsa(c):
    c.data.common.use_msa = False
    recursive_set(c, "max_msa_clusters", 1)
    c.model.extra_msa.enabled = False
    c.loss.masked_msa.weight = 0.0
    c.model.heads.masked_msa.enabled = False
    return c

def fast(c):
    recursive_set(c, "max_msa_clusters", 128)
    recursive_set(c, "max_extra_msa", 512)
    return c


def veryfast(c):
    recursive_set(c, "max_msa_clusters", 32)
    recursive_set(c, "max_extra_msa", 128)
    return c


def close_masked_msa(c):
    c.data.common.masked_msa.profile_prob = 0.0
    c.data.common.masked_msa.same_prob = 0.0
    c.data.common.masked_msa.uniform_prob = 0.0
    c.model.heads.masked_msa.enabled = False
    c.loss.masked_msa.weight = 0.0
    return c

def nomsa_v2(c):
    c = close_masked_msa(c)
    c.data.common.use_msa = False
    c.data.common.use_self_as_extra_msa = True
    recursive_set(c, "max_msa_clusters", 1)
    recursive_set(c, "max_extra_msa", 1)
    c.model.extra_msa.extra_msa_stack.no_col_attention = True
    c.model.evoformer_stack.no_col_attention = True
    return c

def ufconf_af2_v3_c(c):
    c = ufconf_af2_v3_b(c)
    # enhanced time features.
    c.model.use_time_emb_per_layer = False
    # new recycler
    c.model.use_additional_recycler = True
    c.diffusion.chi.enabled = False
    return c

def tordiff(c):
    c.diffusion.chi.enabled = True
    return c