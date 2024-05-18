from typing import *
from unifold.data import residue_constants as rc

import hashlib

import tarfile
import requests
from tqdm import tqdm
import time
import logging
import re
import itertools
import collections
import numpy as np

import os

import pickle, gzip
from unifold.msa import templates, pipeline, parsers
from unifold.msa.tools import hhsearch
from unifold.data.process_multimer import add_assembly_features

from .mmseqs_api import MMseqsRunner

from unifold.dataset import load

logger = logging.getLogger(__name__)

Feature = Dict[str, np.ndarray]

Rotation = Iterable[Iterable]
Translation = Iterable
Operation = Union[str, Tuple[Rotation, Translation]]


def add_hash(x:str, y: str, truncate: int = 5):
    return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:truncate]


def clean_and_validate_sequence(
    input_sequence: str, min_length: int = None, max_length: int = None,
) -> str:
    """Checks that the input sequence is ok and returns a clean version of it."""
    # Remove all whitespaces, tabs and end lines; upper-case.
    clean_sequence = input_sequence.translate(
        str.maketrans('', '', ' \n\t')
    ).upper()
    aatypes = set(rc.restypes_with_x_and_gap)    # 20 standard aatypes + "X-".
    if not set(clean_sequence).issubset(aatypes):
        raise ValueError(
            f'Input sequence contains non-amino acid letters: '
            f'{set(clean_sequence) - aatypes}. AlphaFold only supports 20 standard '
            'amino acids as inputs.')
    if min_length and len(clean_sequence) < min_length:
        raise ValueError(
            f'Input sequence is too short: {len(clean_sequence)} amino acids, '
            f'while the minimum is {min_length}')
    if max_length and len(clean_sequence) > max_length:
        raise ValueError(
            f'Input sequence is too long: {len(clean_sequence)} amino acids, while '
            f'the maximum is {max_length}. You may be able to run it with the full '
            f'Uni-Fold system depending on your resources (system memory, '
            f'GPU memory).')
    return clean_sequence


def validate_input(
    sequences: Sequence[str],
    sequence_ids: Optional[List[str]] = None,
    min_length: int = 6,
    max_length: int = None,
    max_total_length: int = None,
) -> Tuple[Sequence[str], bool]:
    """Validates and cleans input sequences and determines which model to use."""
    sequences = [
        clean_and_validate_sequence(s, min_length, max_length)
        for s in sequences
        if s.strip()
    ]
    if not len(sequences):
        raise ValueError(f'No valid input sequences is found.')

    if sequence_ids is None:
        sequence_ids = [f"I{(i+1):06d}" for i in range(len(sequences))]
    else:
        # clean chars
        sequence_ids = [
            re.sub(r"[^A-Za-z0-9]", '_', sid)
            for sid in sequence_ids
            if sid.strip()
        ]
        # deduplicate
        cnts = collections.defaultdict(int)
        for i, sid in enumerate(sequence_ids):
            cnts[sid] += 1
            if cnts[sid] > 1:
                sequence_ids[i] = f"{sid}_{cnts[sid]}"
        assert len(sequences) == len(sequence_ids), (
            f"incompatible lengths of sequences and ids, "
            f"{len(sequences)} / {len(sequence_ids)}"
        )

    total_length = sum(len(s) for s in sequences)
    if max_total_length and total_length > max_total_length:
        raise ValueError(f'The total length of multimer sequences is too long: '
                         f'{total_length}, while the maximum is '
                         f'{max_total_length}. Please use the full AlphaFold '
                         f'system for long multimers.')
    return sequences, sequence_ids


def sequences_to_a3m(sequences, sequence_ids):
    a3m = "".join(
        f">{sid}\n{seq}\n"
        for sid, seq in zip(sequence_ids, sequences)
    )
    return a3m


def get_null_template(
    query_sequence: Union[List[str], str], num_temp: int = 1
) -> Dict[str, Any]:
    ln = (
        len(query_sequence)
        if isinstance(query_sequence, str)
        else sum(len(s) for s in query_sequence)
    )
    output_templates_sequence = "A" * ln
    output_confidence_scores = np.full(ln, 1.0)

    templates_all_atom_positions = np.zeros(
        (ln, templates.residue_constants.atom_type_num, 3)
    )
    templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
    templates_aatype = templates.residue_constants.sequence_to_onehot(
        output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
    )
    template_features = {
        "template_all_atom_positions": 
        np.tile(
            templates_all_atom_positions[None], [num_temp, 1, 1, 1]
        ),
        "template_all_atom_masks": np.tile(
            templates_all_atom_masks[None], [num_temp, 1, 1]
        ),
        "template_sequence": [f"none".encode()] * num_temp,
        "template_aatype": np.tile(np.array(templates_aatype)[None], [num_temp, 1, 1]),
        "template_domain_names": [f"none".encode()] * num_temp,
        "template_sum_probs": np.zeros([num_temp], dtype=np.float32),
    }
    return template_features


def get_template(
    a3m_lines: str,
    template_path: str,
    query_sequence: str,
    query_index: str = '',
    verbose: bool = False,
    num_null_templates: int = 1,
) -> Dict[str, Any]:
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=template_path,
        max_template_date="2100-01-01",
        max_hits=20,
        kalign_binary_path="kalign",
        release_dates_path=None,
        obsolete_pdbs_path=None,
    )

    hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path="hhsearch", databases=[f"{template_path}/pdb70"]
    )

    hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
    hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
    templates_result = template_featurizer.get_templates(
        query_sequence=query_sequence, hits=hhsearch_hits
    )

    ret = dict(templates_result.features)
    ret["template_all_atom_masks"] = ret.pop("template_all_atom_mask")

    if verbose:
        if len(ret["template_domain_names"]):
            logger.info(
                f"Sequence {query_index} found templates: {ret['template_domain_names'].astype(str).tolist()}"
            )
        else:
            logger.info(
                f"Sequence {query_index} found no templates. Return {num_null_templates} pseudo templates."
            )

    if len(ret["template_domain_names"]) == 0:
        ret = get_null_template(query_sequence)

    return ret


def separate_a3ms(
    directory: str,
    target_a3ms: List[str],
    writeout: bool = True
):
    a3m_chunks = [
        # split packed a3m files by "\x00" for target chunks.
        open(os.path.join(directory, p)).read().strip().split("\x00")[:-1]
        for p in target_a3ms
    ]

    def get_key(a3m: str):
        # first sequence is query sequence.
        return a3m.strip().split('\n')[0][1:].strip()

    results = collections.defaultdict(list)
    for a3m in itertools.chain(*a3m_chunks):
        if len(a3m):
            results[get_key(a3m)].append(a3m)
    results = {
        k: "".join(v)
        for k, v in results.items()
    }
    if writeout:
        for k, v in results.items():
            with open(os.path.join(directory, f"{k}.a3m"), "w") as f:
                f.write(v)
    return results


def parse_template_m8(m8_path):
    def parse_line(line: str):
        p = line.strip().split()
        qid, pdb_id = p[0], p[1]
        return qid, pdb_id
    pairs = [parse_line(l) for l in open(m8_path)]
    template_pdbs = collections.defaultdict(list)
    for qid, pdb_id in pairs:
        template_pdbs[qid].append(pdb_id)
    return dict(template_pdbs)


def make_template_dirs(
    template_pdb_ids: Dict[str, List[str]],
    mmseq_runner: MMseqsRunner,
    output_dir: str,
):
    def process_single(qid, pdb_ids):
        template_dir = os.path.join(output_dir, f"template_{qid}")
        if not os.path.isfile(os.path.join(template_dir, "TEMPL_READY")):
            os.makedirs(template_dir, exist_ok=True)
            mmseq_runner.retrieve_templates(pdb_ids, template_dir)
            os.system(f"cp {template_dir}/pdb70_a3m.ffindex {template_dir}/pdb70_cs219.ffindex")
            os.system(f"touch {template_dir}/pdb70_cs219.ffdata")
            os.system(f"touch {template_dir}/TEMPL_READY")
        return template_dir
    templ_dirs = {
        k: process_single(k, v)
        for k, v in template_pdb_ids.items()
    }
    return templ_dirs


def make_msa_features(
    a3m_lines: List[str],
) -> List[Feature]:
    msa_features = [
        pipeline.make_msa_features(
            [parsers.parse_a3m(a3m, fast=True)]
        ) for a3m in a3m_lines
    ]
    return msa_features


def get_msa_and_template_features(
    sequences: List[str],
    sequence_ids: List[str],
    output_dir: str,
    msa_file_path: Optional[str] = None,
    use_msa: bool = True,
    use_exist_msa: bool = True,
    use_templates: bool = False,
    mmseqs_runner: Optional[MMseqsRunner] = None,
    verbose: bool = False,
    writeout: bool = True,
    **mmseqs_runner_kwargs,
):
    mmseqs_runner = (
        mmseqs_runner
        if mmseqs_runner is not None
        else MMseqsRunner(**mmseqs_runner_kwargs)
    )
    query_a3m = sequences_to_a3m(sequences, sequence_ids)
    if writeout:
        with open(f"{output_dir}/query.a3m", "w") as f:
            f.write(query_a3m)

    seq_features = [
        pipeline.make_sequence_features(seq, sid, len(seq))
        for seq, sid in zip(sequences, sequence_ids)
    ]

    if use_msa or use_templates:
        # run mmseqs2 remote.
        ret = mmseqs_runner.run_mmseqs2(
            query_a3m,
            output_dir,
            endpoint="msa",
            mode="env",
            verbose=verbose,
        )
        if ret != 0:
            raise ValueError(f"MMSeqs2 API unexpectedly exit with return code {ret}.")
        mmseqs2_a3m_dict = separate_a3ms(
            output_dir,
            target_a3ms=["uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"],
            writeout=True,
        )

    trivial_a3m_list = [
            f">{sid}\n{seq}\n"
            for seq, sid in zip(sequences, sequence_ids)
        ]
    if use_msa:
        if use_exist_msa:
            with open(msa_file_path, 'r') as file:
                msa_content = file.read()
            a3m_list = [msa_content]
        else:
            a3m_list = [
                mmseqs2_a3m_dict[sid]
                for sid in sequence_ids
            ]
        msa_features = make_msa_features(a3m_list)
        # create paired msa
        if len(sequences) > 1:
            ret = mmseqs_runner.run_mmseqs2(
                query_a3m,
                output_dir,
                endpoint="pair",
                mode="",
                verbose=verbose,
            )
            if ret != 0:
                raise ValueError(f"MMSeqs2 API for pairing unexpectedly exit with return code {ret}.")
            paired_a3m_lines = separate_a3ms(
                output_dir,
                target_a3ms=["pair.a3m"],
                writeout=False,
            )
            paired_a3m_list = [
                paired_a3m_lines[sid]
                for sid in sequence_ids
            ]
        else:
            paired_a3m_list = trivial_a3m_list
    else:
        a3m_list = paired_a3m_list = trivial_a3m_list

    print("a3m list legth",len(a3m_list[0]))
    msa_features = make_msa_features(a3m_list)
    paired_msa_features = make_msa_features(paired_a3m_list)

    if use_templates:
        m8_path = os.path.join(output_dir, "pdb70.m8")
        if not os.path.isfile(m8_path):
            logger.info("No templates returned from MMSeqs2 API. Use null templates.")
            template_features = [
                get_null_template(seq, 1)
                for seq in sequences
            ]
        else:
            template_pdb_ids = parse_template_m8(m8_path)
            template_paths = make_template_dirs(
                template_pdb_ids,
                mmseqs_runner,
                output_dir
            )
            template_features = [
                get_template(
                    mmseqs2_a3m_dict[sid],
                    template_paths[sid],
                    query_sequence=seq,
                    query_index=sid,
                    verbose=verbose,
                    num_null_templates=1,
                ) for seq, sid in zip(sequences, sequence_ids)
            ]
    else:
        template_features = [
            get_null_template(seq, 1)
            for seq in sequences
        ]

    ret = {
        "feature": seq_features,
        "msa": msa_features,
        "uniprot": paired_msa_features,
        "template": template_features,
    }

    if writeout:
        for key, feats in ret.items():
            for sid, feat in zip(sequence_ids, feats):
                path = f"{output_dir}/{sid}.{key}.pkl.gz"
                pickle.dump(feat, gzip.open(path, 'wb'))

    return ret


def deduplicate_inputs(
    sequences: List[str],
    sequence_ids: List[str],
):
    unique_seqs = []
    unique_seq_ids = []
    id_mapping = {}
    for seq, sid in zip(sequences, sequence_ids):
        if seq not in unique_seqs:
            unique_seqs.append(seq)
            unique_seq_ids.append(sid)
            id_mapping[sid] = sid
        else:
            usid = unique_seq_ids[unique_seqs.index(seq)]
            id_mapping[sid] = usid
    return unique_seqs, unique_seq_ids, id_mapping


def make_input_features(
    output_dir: str,
    sequences: List[str],
    sequence_ids: Optional[List[str]] = None,
    msa_file_path: Optional[str] = None,
    use_msa: bool = True,
    use_exist_msa: bool = False,
    use_templates: bool = False,
    verbose: bool = True,
    # mmseqs api args
    mmseqs_api_url: str = "https://api.colabfold.com",
    retry: int = 3,
    refresh_interval: float = 1,
    timeout: Union[float, str] = "auto",
    # validate args
    min_length: int = 6,
    max_length: int = None,
    max_total_length: int = None,
    # load args
    is_monomer: bool = False,
    load_labels: bool = False,
    use_mmseqs_paired_msa: bool = True,
    symmetry_operations: Optional[List[Operation]] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    # 1. validate inputs.
    seqs, seq_ids = validate_input(
        sequences,
        sequence_ids,
        min_length,
        max_length,
        max_total_length,
    )

    # 2. deduplicate inputs.
    useqs, useq_ids, id_mapping = deduplicate_inputs(seqs, seq_ids)

    all_files_needed = [
        p for p in itertools.chain(
            *[
                [
                    f"{output_dir}/{sid}.{key}.pkl.gz"
                    for sid in useq_ids
                ] for key in ("feature", "msa", "template", "uniprot")
            ]
        )
    ]

    if any(
        not os.path.isfile(p) for p in all_files_needed
    ):  # need to refetch features
        # 3. make features.
        mmseqs_runner = MMseqsRunner(
            mmseqs_api_url,
            retry=retry,
            refresh_interval=refresh_interval,
            timeout=timeout
        )
        results = get_msa_and_template_features(
            useqs,
            useq_ids,
            output_dir,
            msa_file_path=msa_file_path,
            use_msa=use_msa,
            use_exist_msa=use_exist_msa,
            use_templates=use_templates,
            mmseqs_runner=mmseqs_runner,
            verbose=verbose,
            writeout=True,
        )
        del results # reload.

    # 4. load features
    mapped_seq_ids = [id_mapping[sid] for sid in seq_ids]
    if load_labels:
        label_ids = seq_ids
        label_dir = output_dir
    else:
        label_ids = label_dir = None
    features, _ = load(
        sequence_ids=mapped_seq_ids,
        feature_dir=output_dir,
        msa_feature_dir=output_dir,
        template_feature_dir=output_dir,
        uniprot_msa_feature_dir=output_dir,
        label_ids=label_ids,
        label_dir=label_dir,
        is_monomer=is_monomer,
        symmetry_operations = symmetry_operations,
        use_mmseqs_paired_msa=use_mmseqs_paired_msa,
    )

    return seqs, seq_ids, features




