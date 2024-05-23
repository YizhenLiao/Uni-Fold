from unifold.modules.alphafold import *
from unifold.modules.common import Linear, Resnet
from .ufconf_modules import (
    UFConfSM,
    TimeEmbedder,
    ChiAngleEmbedder,
    RelativePositionRecycler,
    UFConfEvoformerStack,
)


class UFConformer(AlphaFold):
    def __init__(self, config):
        super(AlphaFold, self).__init__()

        self.globals = config.globals
        config = config.model
        extra_msa_config = config.extra_msa

        self.input_embedder = InputEmbedder(
            **config["input_embedder"],
            use_chain_relative=config.is_multimer,
        )
        self.recycling_embedder = RecyclingEmbedder(
            **config["recycling_embedder"],
        )

        self.use_chi_embedder = config.chi_embedder.enabled
        if self.use_chi_embedder:
            self.chi_embedder = ChiAngleEmbedder(
                **config["chi_embedder"],
            )

        self.use_additional_recycler = config.use_additional_recycler
        if self.use_additional_recycler:
            self.additional_recycler = RelativePositionRecycler(
                **config["position_recycler"],
            )

        # cancel all template related initialization in AlphaFold.
        assert not config.template.enabled, "must shut template channel."
        if config.extra_msa.enabled:
            self.extra_msa_embedder = ExtraMSAEmbedder(
                **extra_msa_config["extra_msa_embedder"],
            )
            self.extra_msa_stack = ExtraMSAStack(
                **extra_msa_config["extra_msa_stack"],
            )

        self.use_time_emb_per_layer = config.use_time_emb_per_layer
        if not self.use_time_emb_per_layer:
            self.evoformer = EvoformerStack(
                **config["evoformer_stack"],
            )
            self.time_embedder = TimeEmbedder(**config["time_embedder"])
        else:
            self.evoformer = UFConfEvoformerStack(
                d_time=config.time_embedder.d_in,
                time_init=config.time_embedder.init,
                use_time_gated_linear=config.time_embedder.use_gated_linear,
                **config["evoformer_stack"],
            )
            self.time_embedder = None

        self.structure_module = UFConfSM(**config["structure_module"])
        self.aux_heads = AuxiliaryHeads(
            config["heads"],
        )

        self.config = config
        self.dtype = torch.float
        self.inf = self.globals.inf
        if self.globals.alphafold_original_mode:
            self.alphafold_original_mode()

        self.use_x_prev_first = config.use_x_prev_first
        self.use_x_prev_last = config.use_x_prev_last
    
    def iteration_evoformer(self, feats, frames_prev=None):
        # x_prev has new signature, that instead of atom37, provide CA coords only.
        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]

        m, z = self.input_embedder(
            feats["target_feat"],
            feats["msa_feat"],
        )

        if not self.use_time_emb_per_layer:
            m_t, z_t = self.time_embedder(feats["time_feat"])
            m += m_t
            z += z_t

        if self.use_x_prev_first:
            assert frames_prev is not None
            x_prev = frames_prev[..., :3, 3]
            z += self.recycling_embedder.recyle_pos(x_prev)
            if self.use_additional_recycler:
                z += self.additional_recycler(frames_prev, feats["frame_mask"])

        if self.use_chi_embedder:
            chi_emb = self.chi_embedder(
                tf=feats["target_feat"],
                chi_sin_cos=feats["noisy_chi_sin_cos"],
                chi_mask=feats["chi_mask"],
            )
            m += chi_emb[..., None, :, :]

        z += self.input_embedder.relpos_emb(
            feats["residue_index"].long(),
            feats.get("sym_id", None),
            feats.get("asym_id", None),
            feats.get("entity_id", None),
            feats.get("num_sym", None),
        )

        m = m.type(self.dtype)
        z = z.type(self.dtype)
        tri_start_attn_mask, tri_end_attn_mask = gen_tri_attn_mask(pair_mask, self.inf)

        if self.config.template.enabled:    # Never
            template_mask = feats["template_mask"]
            if torch.any(template_mask):
                z = residual(
                    z,
                    self.embed_templates_pair(
                        feats,
                        z,
                        pair_mask,
                        tri_start_attn_mask,
                        tri_end_attn_mask,
                        templ_dim=-4,
                    ),
                    self.training,
                )

        if self.config.extra_msa.enabled:
            a = self.extra_msa_embedder(build_extra_msa_feat(feats))
            extra_msa_row_mask = gen_msa_attn_mask(
                feats["extra_msa_mask"],
                inf=self.inf,
                gen_col_mask=False,
            )
            z = self.extra_msa_stack(
                a,
                z,
                msa_mask=feats["extra_msa_mask"],
                chunk_size=self.globals.chunk_size,
                block_size=self.globals.block_size,
                pair_mask=pair_mask,
                msa_row_attn_mask=extra_msa_row_mask,
                msa_col_attn_mask=None,
                tri_start_attn_mask=tri_start_attn_mask,
                tri_end_attn_mask=tri_end_attn_mask,
            )

        if self.config.template.embed_angles:       # Never
            template_1d_feat, template_1d_mask = self.embed_templates_angle(feats)
            m = torch.cat([m, template_1d_feat], dim=-3)
            msa_mask = torch.cat([feats["msa_mask"], template_1d_mask], dim=-2)

        msa_row_mask, msa_col_mask = gen_msa_attn_mask(
            msa_mask,
            inf=self.inf,
        )

        evoformer_inputs = {
            "msa_mask" : msa_mask,
            "pair_mask" : pair_mask,
            "msa_row_attn_mask" : msa_row_mask,
            "msa_col_attn_mask" : msa_col_mask,
            "tri_start_attn_mask" : tri_start_attn_mask,
            "tri_end_attn_mask" : tri_end_attn_mask,
            "chunk_size" : self.globals.chunk_size,
            "block_size" : self.globals.block_size,
        }

        if not self.use_time_emb_per_layer:
            m, z, s = self.evoformer(m, z, **evoformer_inputs)
        else:
            temb = feats["time_feat"]
            m, z, s = self.evoformer(m, z, temb, **evoformer_inputs)

        return m, z, s
    
    def iteration_evoformer_structure_module(
        self, batch, frames_prev, num_ensembles,
    ):
        z, s = 0, 0
        n_seq = batch["msa_feat"].shape[-3]
        assert num_ensembles >= 1
        for idx in range(num_ensembles):    # num_ensemble = 1 in training
            fetch_cur_batch = lambda t: t[min(t.shape[0] - 1, idx), ...]
            feats = tensor_tree_map(fetch_cur_batch, batch)
            m, z0, s0 = self.iteration_evoformer(feats, frames_prev)
            z += z0
            s += s0
            del z0, s0
        if num_ensembles > 1:
            z /= float(num_ensembles)
            s /= float(num_ensembles)

        outputs = {}

        if self.use_x_prev_last:    # this is rarely used. always use_x_prev_first.
            raise NotImplementedError("ues x prev last has been abandoned.")
            x_prev = frames_prev[..., :3, 3]
            z = z + self.recycling_embedder.recyle_pos(x_prev).type(z.dtype)

        outputs["msa"] = m[..., :n_seq, :, :]
        outputs["pair"] = z
        outputs["single"] = s

        outputs["sm"] = self.structure_module(
            s,
            z,
            feats["time_feat"],
            feats["aatype"],
            feats["noisy_frames"],
            feats["noisy_quats"],
            mask=feats["seq_mask"],
            gen_mask=None,
        )
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["pred_frame_tensor"] = outputs["sm"]["frames"][-1]

        return outputs

    
    def forward(self, batch):
        frames_prev = batch["noisy_frames"][0]
        # noisy frames in [*, L, 4, 4,], so x_prev [*, L, 3]

        num_ensembles = int(batch["msa_mask"].shape[0])

        # convert dtypes in batch
        batch = self.__convert_input_dtype__(batch)
        outputs = self.iteration_evoformer_structure_module(
            batch, frames_prev, num_ensembles,
        )

        if "asym_id" in batch:
            outputs["asym_id"] = batch["asym_id"][0, ...]
        outputs.update(self.aux_heads(outputs))
        return outputs

    def run_structure_module(self, batch, z, s, x_prev = None):
        raise NotImplementedError("depreciated.")
        if self.use_x_prev_last:
            assert x_prev is not None
            z = z + self.recycling_embedder.recyle_pos(x_prev).type(z.dtype)

        out_sm = self.structure_module(
            s,
            z,
            batch["time_feat"],
            batch["aatype"],
            batch["noisy_frames"],
            batch["noisy_quats"],
            mask=batch["seq_mask"],
            gen_mask=None,
        )
        ret_atom_pos = atom14_to_atom37(out_sm["positions"], batch)
        ret_atom_mask = batch["atom37_atom_exists"]
        ret_frames = out_sm["frames"][-1]
        return (ret_atom_pos, ret_atom_mask, ret_frames)

    def __make_input_float__(self):
        self.input_embedder = self.input_embedder.float()
        self.recycling_embedder = self.recycling_embedder.float()
        if not self.use_time_emb_per_layer:
            self.time_embedder = self.time_embedder.float()
        if self.use_additional_recycler:
            self.additional_recycler = self.additional_recycler.float()
        if self.use_chi_embedder:
            self.chi_embedder = self.chi_embedder.float()
