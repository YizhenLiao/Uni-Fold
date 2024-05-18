from unifold.dataset import *
from ufconf.dataset import *
import math

class UFConfClusterDataset(UnicoreDataset):
    def __init__(
        self,
        task,
        args,
        seed,
        config,
        data_path,
        mode="train",
        max_step=None,
        disable_sd=False,
        json_prefix="",
    ):
        assert disable_sd, "must disable sd."
        assert mode == "train", "train implemented only."

        use_v2_weight = config.data.common.use_v2_weight
        self.path = data_path
        chain_weight_suffix = "_60_1,5" if not use_v2_weight else "_85_2,0"
        chain_weights = load_json(
            os.path.join(self.path, "traineval", json_prefix + mode + f"_cluster_sample_weight{chain_weight_suffix}.json")
        )

        global pdb_release_date
        pdb_release_date = load_json(
            os.path.join(self.path, "traineval", json_prefix + "release_date.json")
        )
        max_date = get_datetime(config.data.common.train_max_date)

        def filter_chain_by_date(chain_weights, pdb_release_date, max_date):
            if max_date is None or max_date >= get_datetime("2022-04-30"):
                return chain_weights
            filtered_chain_weights = {}
            filtered_cnt = 0
            for cid, w in chain_weights.items():
                pdb_id = cid.split("_")[0]
                if get_datetime(pdb_release_date[pdb_id][0] <= max_date):
                    filtered_chain_weights[cid] = w
                else:
                    filtered_cnt += 1
            logger.info(
                "Filter out %d chains with release date after %s",
                filtered_cnt,
                max_date,
            )
            return filtered_chain_weights

        if mode == "train":
            chain_weights = filter_chain_by_date(
                chain_weights, pdb_release_date, max_date
            )

        self.chain_to_seq = load_json(
            os.path.join(self.path, "traineval", json_prefix + mode + "_label_to_seq.json")
        )
        seq_len = load_json(
            os.path.join(self.path, "traineval", json_prefix + mode + "_seq_length.json")
        )

        def len_weight(l):
            p1 = max(min(l, 512), 256) / 512
            p2 = l**2 / 1024
            return min(p1, p2)

        def len_weight_v2(l):
            nl = max(min(l, 512), 32) / 512     # [32, 512]
            p = (1-math.cos(math.pi*nl)) / 2    # [0.0096, 1.0]
            return p

        len_weight_fn = len_weight_v2 if use_v2_weight else len_weight
        sample_weights = {
            k: w * len_weight_fn(seq_len[self.chain_to_seq[k]])
            for k, w in chain_weights.items()
        }

        logger.info(f"load {len(sample_weights)} chains.")

        (
            self.feature_path,
            self.msa_feature_path,
            self.template_feature_path,
            self.label_path,
        ) = load_folders(self.path, mode="traineval")

        self.batch_size = (
            args.batch_size
            * distributed_utils.get_data_parallel_world_size()
            * args.update_freq[0]
        )

        if mode == "train":
            self.data_len = max_step * self.batch_size
        else:
            self.data_len = max_step

        self.mode = mode
        self.num_chains, self.chain_ids, self.chain_sample_prob = self.cal_sample_weight(
            sample_weights
        )

        self.config = config.data
        self.seed = seed

        # diffold dataset init
        self.diffusion_config = config.diffusion
        self.task = task
        self.diffuser = Diffuser(config.diffusion)

    def cal_sample_weight(self, sample_weight):
        prot_keys = list(sample_weight.keys())
        sum_weight = sum(sample_weight.values())
        sample_prob = [sample_weight[k] / sum_weight for k in prot_keys]
        num_prot = len(prot_keys)
        return num_prot, prot_keys, sample_prob

    def sample_chain(self, idx):
        if self.mode == "train":
            with data_utils.numpy_seed(self.seed, idx, key="data_sample"):
                seq_idx = np.random.choice(self.num_chains, p=self.chain_sample_prob)
                chain_id = self.chain_ids[seq_idx]
        else:
            seq_idx = idx % self.num_chains
            chain_id = self.chain_ids[seq_idx]
        return chain_id

    def __getitem__(self, idx):
        chain_id = self.sample_chain(idx)
        sequence_id = self.chain_to_seq[chain_id]
        feature_path, msa_feature_path, template_feature_path, label_path = (
            self.feature_path,
            self.msa_feature_path,
            self.template_feature_path,
            self.label_path,
        )
        features, _ = load_and_process(
            self.config,
            self.mode,
            self.seed,
            batch_idx=(idx // self.batch_size),
            data_idx=idx,
            is_distillation=False,
            sequence_ids=[sequence_id],
            feature_dir=feature_path,
            msa_feature_dir=msa_feature_path,
            template_feature_dir=template_feature_path,
            uniprot_msa_feature_dir=None,
            label_ids=[chain_id],
            label_dir=label_path,
            symmetry_operations=None,
            is_monomer=True,
        )
        with numpy_seed(self.seed, idx, key="get_diffusion_seed"):
            diffusion_seed = np.random.randint(1 << 31)
        features = diffuse_inputs(
            features, self.diffuser, diffusion_seed, self.diffusion_config, task=self.task
        )
        return features


    def __len__(self):
        return self.data_len

    @staticmethod
    def collater(samples):
        return data_utils.collate_dict(samples, dim=1)
