import torch
import torch.nn as nn
import torch.utils.checkpoint

from unifold.modules.evoformer import EvoformerStack, SimpleModuleList
from unifold.modules.common import Linear, Resnet, ResnetBlock
from unifold.modules.frame import Frame
from .sm import DiffoldSM, GatedLinear


class UFConfEvoformerStack(EvoformerStack):
    def __init__(self, d_time, time_init="default", use_time_gated_linear=True, **kwargs):
        super().__init__(**kwargs)
        linear_cls = GatedLinear if use_time_gated_linear else Linear
        self.linears_t_m = SimpleModuleList([
            linear_cls(d_time, kwargs["d_msa"], init=time_init)
            for _ in range(len(self.blocks))
        ])
        self.linears_t_zi = SimpleModuleList([
            linear_cls(d_time, kwargs["d_pair"], init=time_init)
            for _ in range(len(self.blocks))
        ])
        self.linears_t_zj = SimpleModuleList([
            linear_cls(d_time, kwargs["d_pair"], init=time_init)
            for _ in range(len(self.blocks))
        ])

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        **inputs
    ):
        t = t.type_as(m)
        for b, t_lin, zi_lin, zj_lin in zip(
            self.blocks, self.linears_t_m, self.linears_t_zi, self.linears_t_zj,
        ):
            m = m + t_lin(t)[..., None, :, :]
            z = z + zi_lin(t)[..., None, :, :]
            z = z + zj_lin(t)[..., :, None, :]

            wrapped_forward = lambda _m, _z: b(_m, _z, **inputs)
            if torch.is_grad_enabled():
                m, z = torch.utils.checkpoint.checkpoint(wrapped_forward, m, z)
            else:
                m, z = wrapped_forward(m, z)

        assert not self._is_extra_msa_stack
        seq_dim = -3
        index = torch.tensor([0], device=m.device)
        s = self.linear(torch.index_select(m, dim=seq_dim, index=index))
        s = s.squeeze(seq_dim)

        return m, z, s


class TimeEmbedder(nn.Module):
    def __init__(self, d_in, d_msa, d_pair, use_gated_linear, init="default", **base_args):
        super().__init__(**base_args)
        linear_cls = GatedLinear if use_gated_linear else Linear
        self.to_msa = linear_cls(d_in=d_in, d_out=d_msa, init=init)
        self.to_pair_i = linear_cls(d_in=d_in, d_out=d_pair, init=init)
        self.to_pair_j = linear_cls(d_in=d_in, d_out=d_pair, init=init)

    def forward(self, x):
        '''
        * L Dt -> * 1 L Dm, * L L Dz
        '''
        m = self.to_msa(x)[..., None, :, :]
        z = self.to_pair_i(x)[..., None, :, :] + self.to_pair_j(x)[..., :, None, :]
        return m, z


def make_rbf_bins(min_bin, max_bin, num_bins):
    intervals = torch.linspace(min_bin, max_bin, num_bins, requires_grad=False)
    return intervals


class RelativePositionRecycler(nn.Module):
    def __init__(
        self,
        d_pair,
        d_hid,
        num_blocks, 
        cutoff: float,
        num_bins: int,
        sigma: float = None,
        init="default", **base_args
    ):
        super().__init__(**base_args)
        self.cutoff = cutoff
        self.rbf_bins = make_rbf_bins(-cutoff, cutoff, num_bins)
        self.sigma = sigma or 2*cutoff/(num_bins-1)
        self.use_resnet = num_blocks > 0
        d_in = 3 * num_bins + 1
        if self.use_resnet:
            self.model = Resnet(d_in, d_pair, d_hid, num_blocks, final_init=init)
        else:
            self.model = Linear(d_in, d_pair, init=init)

    @property
    def dtype(self):
        if not self.use_resnet:
            return self.model.weight.dtype
        else:
            return self.model.linear_in.weight.dtype

    def forward(self, frames, frame_mask):
        '''
        copying recycle_embedder.recycle_pos, but with renewed model / params.
        '''
        self.rbf_bins = self.rbf_bins.type_as(frames)
        # get relpos feats
        frames = Frame.from_tensor_4x4(frames)
        relpos = frames[..., None].invert_apply(frames.get_trans()[..., None, :, :])
        relpos = relpos.clamp(min=-self.cutoff, max=self.cutoff)
        rbd = relpos[..., None] - self.rbf_bins
        rbf = torch.exp(-.5 * (rbd/self.sigma)**2)
        rbf = rbf.view(*rbf.shape[:-2], -1)     # * L L 3*nb
        # handle missing pairs
        is_missing = frame_mask[..., None] * frame_mask[..., None, :]   # * L L
        is_missing = is_missing[..., None]      # * L L 1
        rbf *= is_missing
        rbf = torch.cat([rbf, is_missing], dim=-1).type(self.dtype)
        # forward model
        z = self.model(rbf)
        return z



class ChiAngleEmbedder(nn.Module):
    def __init__(self, d_in, d_hid, d_out, num_blocks, init="default", enabled=True, **base_args):
        super().__init__(**base_args)
        self.exclude_tf_0 = (d_in == 33)
        self.linear_in = GatedLinear(d_in, d_hid, init="default")
        self.act = nn.GELU()
        self.resnet_blocks = SimpleModuleList([
            ResnetBlock(d_hid) for _ in range(num_blocks)
        ])
        self.linear_out = Linear(d_hid, d_out, init=init)

    def forward(self, tf, chi_sin_cos, chi_mask):
        '''
        * L Dt -> * 1 L Dm, * L L Dz
        '''
        chi_sin_cos *= chi_mask[..., None]      # set all masked values as 0 (not necessary because addnoise already did so, but this is safe)
        chi_feat = torch.cat([chi_sin_cos, chi_mask[..., None]], dim=-1)    # * L 4 2+1
        chi_feat = chi_feat.view(*chi_feat.shape[:-2], -1)  # * L 12
        if self.exclude_tf_0:
            # align with input embedder
            tf = tf[..., 1:]
        x = torch.cat([tf, chi_feat], dim=-1)   # * L 33

        x = self.linear_in(x)
        for b in self.resnet_blocks:
            x = b(x)
        x = self.linear_out(x)
        return x
