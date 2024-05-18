from unifold.task import *
from .dataset import DiffoldDataset, DiffoldMultimerDataset
from .dataset_clu import UFConfClusterDataset

@register_task("ufconf")
class DiffoldTask(AlphafoldTask):
    @staticmethod
    def add_args(parser):
        AlphafoldTask.add_args(parser)
        parser.add_argument("--use-multimer", action="store_true")
        parser.add_argument("--use-cluster-dataset", action="store_true")

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if self.args.use_multimer:
            assert not self.args.use_cluster_dataset
            train_data_class = eval_data_class = DiffoldMultimerDataset
        else:
            if self.args.use_cluster_dataset:
                assert self.args.disable_sd
                train_data_class = UFConfClusterDataset
                eval_data_class = DiffoldDataset
            else:
                train_data_class = eval_data_class = DiffoldDataset

        if split == "train":
            dataset = train_data_class(
                split,
                self.args,
                self.args.seed + 81,
                self.config,
                self.args.data,
                mode="train",
                max_step=self.args.max_update,
                disable_sd=self.args.disable_sd,
                json_prefix=self.args.json_prefix,
            )
        else:
            dataset = eval_data_class(
                split,
                self.args,
                self.args.seed + 81,
                self.config,
                self.args.data,
                mode="eval",
                max_step=128,
                disable_sd=True,
                json_prefix=self.args.json_prefix,
            )

        self.datasets[split] = dataset