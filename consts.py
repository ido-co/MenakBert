from pathlib import Path

import torch

DATA_MODULES_CACHE = f"data_modules_cache.pkl"

BACKBONE_NAME = "tavbert"
BACKBONE = "tau/tavbert-he"
BACKBONE_MODEL_LOCAL_URL = "https://drive.google.com/drive/folders/1K78B5SM8FjBc_5r-UWTwoj1x105xpksK?usp=sharing"

PROJ_FOLDER = Path.cwd()
TRAIN_PATH = PROJ_FOLDER / r"hebrew_diacritized/data/train"
VAL_PATH = PROJ_FOLDER / "hebrew_diacritized/data/validation"
TEST_PATH = PROJ_FOLDER / "hebrew_diacritized/data/test"

train_data = [TRAIN_PATH]
val_data = [VAL_PATH]
test_data = [TEST_PATH]

Val_BatchSize = 32
Train_BatchSize = 32

DROPOUT = 0.1

LR = 1e-5

MAX_EPOCHS = 100
MIN_EPOCHS = 5

MAX_LEN = 100
MIN_LEN = 10

LINEAR_LAYER_SIZE = 2048

DEFAULT_CFG = {
    "train_data": TRAIN_PATH,
    "val_data": VAL_PATH,
    "test_data": TEST_PATH,
    "model": BACKBONE,
    "maxlen": MAX_LEN,
    "minlen": MIN_LEN,
    "lr": LR,
    "dropout": DROPOUT,
    "train_batch_size": Train_BatchSize,
    "val_batch_size": Val_BatchSize,
    "max_epochs": MAX_EPOCHS,
    "min_epochs": MIN_EPOCHS,
    "linear_layer_size": LINEAR_LAYER_SIZE,
    "weighted_loss": True,
    "split_sentence": False  # TODO check
}
N_CLASSES = ["none", "rafa", "shva", "hataph segol",
             "hataph patah", "hataph kamats",
             "hirik", "chere", "segol", "phatah",
             "kamats", "hulam", "kubuch", "shuruk"]
S_CLASSES = ["NONE", "mask", "sin", "shin"]
D_CLASSES = ["NONE", "RAFE", "DAGESH"]

# TODO check what is the right why to do so
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# weights for all 4 dm
ALL_WEIGHTS = \
    [
        None,  # for religion
        None,  # for pre_modern
        # for early_modern
        {'N': torch.tensor([2.4227e-01, 2.5790e-01, 9.7148e-02, 8.7762e-04, 1.2099e-02, 1.5516e-04,
                            7.8204e-02, 2.8595e-02, 4.4660e-02, 7.7416e-02, 8.1133e-02, 4.8025e-02,
                            7.1316e-04, 3.0812e-02], device=device),
         'S': torch.tensor([9.6230e-01, 1.5810e-04, 3.4004e-02, 3.5340e-03], device=device),
         'D': torch.tensor([0.3800, 0.5218, 0.0982], device=device)},
        # for modern
        {'N': torch.tensor([2.4920e-01, 2.6323e-01, 8.8410e-02, 1.0692e-03, 1.2552e-02, 1.4649e-04,
                            7.2875e-02, 2.8470e-02, 4.6895e-02, 7.4532e-02, 8.5893e-02, 4.8047e-02,
                            2.6159e-05, 2.8658e-02], device=device),
         'S': torch.tensor([9.6249e-01, 3.2961e-04, 3.4374e-02, 2.8073e-03], device=device),
         'D': torch.tensor([0.3914, 0.5126, 0.0961], device=device)}
    ]
