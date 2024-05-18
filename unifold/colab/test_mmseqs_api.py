from unifold.colab import make_input_features
from unifold.config import model_config

import sys
import logging
logging.basicConfig(level=logging.INFO)
output_dir = sys.argv[1]

seqs = [
    "LILNLRGGAFVSNTQITMADKQKKFINEIQEGDLVRSYSITDETFQQNAVTSIVKHEADQLCQINFGKQH",
    "VVCTVNHRFYDPESKLWKSVCPHPGSGISFLKKLLLYDYLLSEEGEKLQITE",
    "IKTFTTKQPVFIYHIQVENNHNFFANGVLAHAMQVSI"
]

features = make_input_features(
    "test",
    output_dir,
    seqs,
    sequence_ids=None,
    use_msa=True,
    use_templates=True,
    verbose=True
)

from unifold.diffold.utils import recur_print
recur_print(features)
