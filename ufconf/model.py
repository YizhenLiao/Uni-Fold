from unifold.model import *
from .diffold import Diffold
from .config import model_config

@register_model("ufconf")
class DiffoldModel(AlphafoldModel):
    def __init__(self, args):
        super(AlphafoldModel, self).__init__()  # for BaseUnicoreModel
        base_architecture(args)
        self.args = args
        config = model_config(
            self.args.model_name,
            train=True,
        )
        self.model = Diffold(config)
        self.config = config

@register_model_architecture("ufconf", "ufconf")
def base_architecture(args):
    args.model_name = getattr(args, "model_name", "ufconf_af2_v3")
