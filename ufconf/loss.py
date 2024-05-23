from unifold.loss import *
from unifold.losses.utils import masked_mean

def update_penalty(
    update_vec: torch.Tensor,
    frame_mask: torch.Tensor,
    loss_dict: dict,
    eps: float,
):
    norm = torch.sqrt(torch.sum(update_vec ** 2, dim=-1) + eps)
    loss = masked_mean(frame_mask[None], norm, dim=-1, eps=eps).mean(dim=0)
    loss_dict["update_norm"] = loss.data
    return loss


def global_atom_error(
    sidechain_atom_pos: torch.Tensor,   # s * L 14 3
    renamed_atom14_gt_positions: torch.Tensor,  # * L 14 3
    renamed_atom14_gt_exists: torch.Tensor, # * L 14
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
    loss_dict: dict = {},
    **kwargs,
) -> torch.Tensor:
    err = torch.sqrt(
        torch.sum(
            sidechain_atom_pos - renamed_atom14_gt_positions[None], dim=-1
        ) ** 2 + eps
    )   # s * L 14
    err = err.clamp_max(clamp_distance) / length_scale
    err = masked_mean(renamed_atom14_gt_exists[None], err, dim=(-1, -2), eps=eps)
    err = err.mean(dim=0)
    loss_dict["global_atom_error"] = err.data
    return err


@register_loss("ufconf")
class UFConfLoss(AlphafoldLoss):
    # TODO: see how we weight losses among different diffusion steps.
    @staticmethod
    def forward(model, batch, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # return config in model.
        out, config = model(batch)
        num_recycling = batch["msa_feat"].shape[0]
        
        # remove recyling dim
        batch = tensor_tree_map(lambda t: t[-1, ...], batch)
        
        loss, sample_size, logging_output = UFConfLoss.loss(out, batch, config)
        logging_output["num_recycling"] = num_recycling
        return loss, sample_size, logging_output

    @staticmethod
    def loss(out, batch, config):

        if "violation" not in out.keys() and config.violation.weight:
            out["violation"] = find_structural_violations(
                batch, out["sm"]["positions"], **config.violation)

        if "renamed_atom14_gt_positions" not in out.keys():
            batch.update(
                compute_renamed_ground_truth(batch, out["sm"]["positions"]))
        
        diffusion_t = batch["diffusion_t"].float()

        loss_dict = {}
        loss_fns = {
            "chain_centre_mass": lambda: chain_centre_mass_loss(
                pred_atom_positions=out["final_atom_positions"],
                true_atom_positions=batch["all_atom_positions"],
                atom_mask=batch["all_atom_mask"],
                asym_id=batch["asym_id"],
                **config.chain_centre_mass,
                loss_dict=loss_dict,
            ),
            "distogram": lambda: distogram_loss(
                logits=out["distogram_logits"],
                pseudo_beta=batch["pseudo_beta"],
                pseudo_beta_mask=batch["pseudo_beta_mask"],
                **config.distogram,
                loss_dict=loss_dict,
            ),
            "experimentally_resolved": lambda: experimentally_resolved_loss(    # not used
                logits=out["experimentally_resolved_logits"],
                atom37_atom_exists=batch["atom37_atom_exists"],
                all_atom_mask=batch["all_atom_mask"],
                resolution=batch["resolution"],
                **config.experimentally_resolved,
                loss_dict=loss_dict,
            ),
            "fape": lambda: fape_loss(
                out,
                batch,
                config.fape,
                loss_dict=loss_dict,
            ),
            "masked_msa": lambda: masked_msa_loss(
                logits=out["masked_msa_logits"],
                true_msa=batch["true_msa"],
                bert_mask=batch["bert_mask"],
                loss_dict=loss_dict,
            ),
            "pae": lambda: pae_loss(
                logits=out["pae_logits"],
                pred_frame_tensor=out["pred_frame_tensor"],
                true_frame_tensor=batch["true_frame_tensor"],
                frame_mask=batch["frame_mask"],
                resolution=batch["resolution"],
                **config.pae,
                loss_dict=loss_dict,
            ),
            "plddt": lambda: plddt_loss(
                logits=out["plddt_logits"],
                all_atom_pred_pos=out["final_atom_positions"],
                all_atom_positions=batch["all_atom_positions"],
                all_atom_mask=batch["all_atom_mask"],
                resolution=batch["resolution"],
                **config.plddt,
                loss_dict=loss_dict,
            ),
            "repr_norm": lambda: repr_norm_loss(    # not used
                out["delta_msa"],
                out["delta_pair"],
                out["msa_norm_mask"],
                batch["pseudo_beta_mask"],
                **config.repr_norm,
                loss_dict=loss_dict,
            ),
            "supervised_chi": lambda: supervised_chi_loss(
                pred_angles_sin_cos=out["sm"]["angles"],
                pred_unnormed_angles_sin_cos=out["sm"]["unnormalized_angles"],
                true_angles_sin_cos=batch["chi_angles_sin_cos"],
                aatype=batch["aatype"],
                seq_mask=batch["seq_mask"],
                chi_mask=batch["chi_mask"],
                **config.supervised_chi,
                loss_dict=loss_dict,
            ),
            "violation": lambda: violation_loss(
                out["violation"],
                loss_dict=loss_dict,
                bond_angle_loss_weight=config.violation.bond_angle_loss_weight,
            ),
            "update_penalty": lambda: update_penalty(   # new
                out["sm"]["update_vec"],
                batch["seq_mask"],
                loss_dict=loss_dict,
                eps=config.update_penalty.eps,
            ),
            "global_atom_error": lambda: global_atom_error( # new
                out["sm"]["positions"],
                **batch, **config.global_atom_error, loss_dict=loss_dict,
            ),
        }

        cum_loss = 0
        bsz = batch["seq_mask"].shape[0]
        with torch.no_grad():
            seq_len = torch.sum(batch["seq_mask"].float(), dim=-1)
            seq_length_weight = seq_len**0.5
        
        assert (
            len(seq_length_weight.shape) == 1 and seq_length_weight.shape[0] == bsz
        ), seq_length_weight.shape
        
        for loss_name, loss_fn in loss_fns.items():
            weight = config[loss_name].weight
            time_scaling = config[loss_name].get("time_scaling", 1.)
            if weight > 0.:
                loss = loss_fn()
                # always use float type for loss
                assert loss.dtype == torch.float, loss.dtype
                assert len(loss.shape) == 1 and loss.shape[0] == bsz, (loss_name, loss.shape)

                if any(torch.isnan(loss)) or any(torch.isinf(loss)):
                    logging.warning(f"{loss_name} loss is NaN. Skipping...")
                    loss = loss.new_tensor(0.0, requires_grad=True)

                diffusion_t = diffusion_t ** config.time_scaling_coef_a
                if config["time_scaling_method"] == "linear":
                    time_scaling_weight = 1. - diffusion_t * (1. - time_scaling)
                elif config["time_scaling_method"] == "exp":
                    time_scaling_weight = time_scaling ** diffusion_t

                scaled_loss = time_scaling_weight * loss
                if time_scaling != 1:
                    loss_dict["scaled_" + loss_name] = scaled_loss.data
                cum_loss = cum_loss + weight * scaled_loss

        for key in loss_dict:
            loss_dict[key] = float((loss_dict[key]).mean())

        loss = (cum_loss * seq_length_weight).mean()

        logging_output = loss_dict
        # sample size fix to 1, so the loss (and gradients) will be averaged on all workers.
        sample_size = 1
        logging_output["loss"] = loss.data
        logging_output["bsz"] = bsz
        logging_output["sample_size"] = sample_size
        logging_output["seq_len"] = seq_len
        logging_output["t"] = batch["diffusion_t"]
        # logging_output["num_recycling"] = num_recycling
        return loss, sample_size, logging_output
