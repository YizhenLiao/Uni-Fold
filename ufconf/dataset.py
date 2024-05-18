
from typing import Optional
from ml_collections import ConfigDict
from unicore.data.data_utils import numpy_seed
from unifold.dataset import *
from .diffuse_ops import diffuse_inputs
from .diffusion.diffuser import Diffuser

from Bio.PDB import PDBParser
from collections import defaultdict
import string
import re

from unifold.data import residue_constants as rc
parser = PDBParser(QUIET=True)
seq_map_trans = str.maketrans(
    ''.join(rc.restypes_with_x),
    ''.join(chr(v) for v in range(len(rc.restypes_with_x)))
)

NumpyDict = Dict[str, np.ndarray]
Reference = Dict[str, Dict[str, NumpyDict]] # chain_id + res_id to numpy dict


class DiffoldDataset(UnifoldDataset):
    def __init__(self, task, args, seed, config, data_path, mode="train", max_step=None, disable_sd=False, json_prefix=""):
        super().__init__(args, seed, config, data_path, mode, max_step, disable_sd, json_prefix)
        self.diffusion_config = config.diffusion
        self.task = task
        self.diffuser = Diffuser(config.diffusion)

    def __getitem__(self, idx):
        features = super().__getitem__(idx)
        with numpy_seed(self.seed, idx, key="get_diffusion_seed"):
            diffusion_seed = np.random.randint(1 << 31)
        features = diffuse_inputs(
            features, self.diffuser, diffusion_seed, self.diffusion_config, task=self.task
        )
        return features


class DiffoldMultimerDataset(UnifoldMultimerDataset):
    def __init__(self, task, args, seed, config, data_path, mode="train", max_step=None, disable_sd=False, json_prefix=""):
        super().__init__(args, seed, config, data_path, mode, max_step, disable_sd, json_prefix)
        self.diffusion_config = config.diffusion
        self.task = task
        self.diffuser = Diffuser(config.diffusion)

    def __getitem__(self, idx):
        features, _ = super().__getitem__(idx)
        with numpy_seed(self.seed, idx, key="get_diffusion_seed"):
            diffusion_seed = np.random.randint(1 << 31)
        features = diffuse_inputs(
            features, self.diffuser, diffusion_seed, self.diffusion_config, task=self.task
        )
        return features

    @staticmethod
    def collater(samples):
        # first dim is recyling. bsz is at the 2nd dim
        return data_utils.collate_dict(samples, dim=1)
    

import urllib
import io
from unifold.data import residue_constants

def get_pdb(filename):
    if not os.path.exists(filename):
        print('pdb not exist, download ... ')
        pdb_id = filename.split('/')[-1].split('.')[0]
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        urllib.request.urlretrieve(url, filename)
        print(f'finish download {pdb_id}')
    return filename

def read_pdb(path, line_list=None):
    with open(path, 'r') as f:
        if line_list:
            return  f.readlines()
        else: return f.read() 

def load_pdb_feat(filename):
    pdb_file = get_pdb(filename)
    return from_pdb_string(read_pdb(pdb_file))


# add heteratom ?
def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None, No_model=0) -> dict:
    """Takes a PDB string and constructs a Protein object.
    #### NOTICE: the difference between `unifold/data/protein.py` is that here we return dict type.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If chain_id is specified (e.g. A), then only that chain
        is parsed. Otherwise all chains are parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        if No_model == 0:
            logger.warn(f'found {len(models)} models in the pdb, and using default model {No_model}, pease make sure you want No. `{No_model}` model in the pdb files.')
    model = models[No_model]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    res_list_idx = []
    chain_ids = []
    b_factors = []
    res_shortnames = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res_idx, res in enumerate(chain):
            if res.id[2] != " ":
                raise ValueError(
                    f"PDB contains an insertion code at chain {chain.id} and residue " +
                    f"index {res.id[1]}. These are not supported."
                )
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            res_shortnames.append(res_shortname)
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            # if np.sum(mask) < 0.5:    # keep
            #     # If no known atom positions are reported for the residue then skip it.
            #     continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            res_list_idx.append(res_idx)
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])
    chain_index_map = [(c, r, i) for c,r,i in zip(chain_ids, residue_index, res_list_idx)]

    return {
        'all_atom_positions' : np.array(atom_positions),
        'all_atom_mask' : np.array(atom_mask),
        'aatype' : np.array(aatype),
        'residue_index' : np.array(residue_index),
        'chain_index' : chain_index,
        'pdb_idx' : chain_index_map,
        'b_factors' : np.array(b_factors),
        'res_seq' : np.array(res_shortnames)
    }

import os
import urllib.request
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import io

def get_cif(filename):
    if not os.path.exists(filename):
        print('CIF file does not exist, downloading...')
        pdb_id = filename.split('/')[-1].split('.')[0]
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        urllib.request.urlretrieve(url, filename)
        print(f'Finished downloading {pdb_id}')
    return filename

def load_cif_feat(filename,chain_id: Optional[str] = None):
    cif_file = get_cif(filename)
    mmcif_dict = MMCIF2Dict(cif_file)
    
    parser = MMCIFParser()
    structure = parser.get_structure("none", cif_file)
    model = next(structure.get_models())  # Assuming interest in the first model
    
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    res_list_idx = []
    chain_ids = []
    b_factors = []
    res_shortnames = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res_idx, res in enumerate(chain):
            # if res.id[2] != " ":
            #     raise ValueError(
            #         f"PDB contains an insertion code at chain {chain.id} and residue " +
            #         f"index {res.id[1]}. These are not supported."
            #     )
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            res_shortnames.append(res_shortname)
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            # if np.sum(mask) < 0.5:    # keep
            #     # If no known atom positions are reported for the residue then skip it.
            #     continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            res_list_idx.append(res_idx)
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])
    chain_index_map = [(c, r, i) for c,r,i in zip(chain_ids, residue_index, res_list_idx)]

    return {
        'all_atom_positions' : np.array(atom_positions),
        'all_atom_mask' : np.array(atom_mask),
        'aatype' : np.array(aatype),
        'residue_index' : np.array(residue_index),
        'chain_index' : chain_index,
        'pdb_idx' : chain_index_map,
        'b_factors' : np.array(b_factors),
        'res_seq' : np.array(res_shortnames)
    }