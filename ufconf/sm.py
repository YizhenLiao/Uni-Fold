from unifold.modules.structure_module import *
from unifold.modules.frame import rot_to_quat


class GatedLinear(nn.Module):
    def __init__(self, d_in, d_out, init="default"):
        super().__init__()
        self.linear_gate = Linear(d_in, d_out, init="gating")
        self.linear_out = Linear(d_in, d_out, init=init)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        g = self.linear_gate(x)
        g = self.sigmoid(g)
        o = g * self.linear_out(x)
        return o


class DiffoldSM(StructureModule):
    def __init__(self, d_time, d_single, d_pair, d_ipa, d_angle, num_heads_ipa, num_qk_points, num_v_points, dropout_rate, num_blocks, no_transition_layers, num_resnet_blocks, num_angles, trans_scale_factor, separate_kv, ipa_bias, epsilon, inf, **kwargs):
        super().__init__(d_single, d_pair, d_ipa, d_angle, num_heads_ipa, num_qk_points, num_v_points, dropout_rate, num_blocks, no_transition_layers, num_resnet_blocks, num_angles, trans_scale_factor, separate_kv, ipa_bias, epsilon, inf, **kwargs)
        self.gate_lin_time = GatedLinear(d_time, d_single, init="final")

    def forward(
        self,
        s,
        z,
        time_feat,
        aatype,
        frames,
        quats,
        mask,
        gen_mask=None,
    ):
        # generate square mask
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = gen_attn_mask(square_mask, -self.inf).unsqueeze(-3)
        s = s + self.gate_lin_time(time_feat.type_as(s))
        s = self.layer_norm_s(s)
        z = self.layer_norm_z(z)
        initial_s = s
        s = self.linear_in(s)

        frames = frames.type_as(s)    # to bf16 after getting quat.
        quats = quats.type_as(s)

        backb_to_global = Frame.from_tensor_4x4(
            frames
        ).scale_translation(1. / self.trans_scale_factor)

        quat_encoder = Quaternion.from_tensor_7(quats).scale_translation(1. / self.trans_scale_factor)

        outputs = []
        for i in range(self.num_blocks):
            s = residual(s, self.ipa(s, z, backb_to_global, square_mask), self.training)
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            update_vec = self.bb_update(s)
            if gen_mask is not None:
                update_vec *= gen_mask[..., None]

            # update quaternion encoder
            # use backb_to_global to avoid quat-to-rot conversion
            quat_encoder = quat_encoder.compose_update_vec(
                update_vec, pre_rot_mat=backb_to_global.get_rots()
            )

            # initial_s is always used to update the backbone
            unnormalized_angles, angles = self.angle_resnet(s, initial_s)

            # convert quaternion to rotation matrix
            backb_to_global = Frame(
                Rotation(
                    mat=quat_encoder.get_rot_mats(),
                ),
                quat_encoder.get_trans(),
            )
            if i == self.num_blocks - 1:
                all_frames_to_global = self.torsion_angles_to_frames(
                    backb_to_global.scale_translation(self.trans_scale_factor),
                    angles,
                    aatype,
                )

                pred_positions = self.frames_and_literature_positions_to_atom14_pos(
                    all_frames_to_global,
                    aatype,
                )

            preds = {
                "update_vec": update_vec,
                "frames": backb_to_global.scale_translation(
                    self.trans_scale_factor
                ).to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
            }

            outputs.append(preds)
            if i < (self.num_blocks - 1):
                # stop gradient in iteration
                quat_encoder = quat_encoder.stop_rot_gradient()
                backb_to_global = backb_to_global.stop_rot_gradient()

        outputs = dict_multimap(torch.stack, outputs)
        outputs["sidechain_frames"] = all_frames_to_global.to_tensor_4x4()
        outputs["positions"] = pred_positions
        outputs["single"] = s

        return outputs
