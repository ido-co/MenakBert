import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import gdown
import torch
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer

from HebrewDataModule import create_data_modules
from MenakBert import MenakBert
from consts import BACKBONE, DEFAULT_CFG, BACKBONE_NAME, ALL_WEIGHTS, BACKBONE_MODEL_LOCAL_URL, \
    MODEL_CHECKPOINT_FILENAME, MODEL_CHECKPOINT_DIR_PATH, CSV_HEAD, OUTPUT_FOLDER
from evaluation import compare_by_file_from_checkpoint
from metrics import all_stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_weights(train_data):
    # TODO when refactoring move to a better place, might be in utils or Data Module
    tmp = train_data.counter['N']
    n_weights = torch.tensor([tmp[i] for i in range(len(tmp.keys()))])
    n_weights = (n_weights / n_weights.sum()).to(device)

    tmp = train_data.counter['D']
    d_weights = torch.tensor([tmp[i] for i in range(len(tmp.keys()))])
    d_weights = (d_weights / d_weights.sum()).to(device)

    tmp = train_data.counter['S']
    s_weights = torch.tensor([tmp[i] for i in range(len(tmp.keys()))])
    s_weights = (s_weights / s_weights.sum()).to(device)

    weights = {'N': n_weights, 'S': s_weights, 'D': d_weights}
    return weights


def download_backbone_model_essentials():
    # TODO ask ido about the name, maybe there is a better convention
    # TODO move it to tavbert module, we should download this data dynamically
    # TODO we need to add a flag and according to its value we should usr the local version or to download it from web
    if not os.path.exists(BACKBONE_NAME):
        os.mkdir(BACKBONE_NAME)
        gdown.download_folder(BACKBONE_MODEL_LOCAL_URL, output=BACKBONE_NAME)


def calculate_warmup_steps(params, dm):
    steps_per_epoch = len(dm.train_data) // params['train_batch_size']
    total_training_steps = steps_per_epoch * params['max_epochs']
    warmup_steps = total_training_steps // 5
    return warmup_steps, total_training_steps


def setup_model(warmup_steps, total_training_steps, lr, dropout, linear_size, weights=None):
    # TODO ask ido if it should be part of MenakBert
    download_backbone_model_essentials()

    model = MenakBert(backbone=BACKBONE,
                      dropout=dropout,
                      lr=lr,
                      linear_size=linear_size,
                      n_warmup_steps=warmup_steps,
                      n_training_steps=total_training_steps,
                      weights=weights,
                      trainable=True)
    return model


def setup_trainer(max_epochs):
    # config training
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_CHECKPOINT_DIR_PATH,
        filename=MODEL_CHECKPOINT_FILENAME,
        save_top_k=1,
        verbose=True,
        monitor="f1_score_N",
        mode="max"
    )

    logger = TensorBoardLogger(OUTPUT_FOLDER / "lightning_logs", name="nikkud_logs")
    early_stopping_callback = EarlyStopping(monitor='train_loss', patience=5)

    save_model_callback = SaveModelCallback(base_directory=OUTPUT_FOLDER / "huggingface_models")

    trainer = Trainer(
        logger=logger,
        # auto_lr_find=True,
        callbacks=[checkpoint_callback, save_model_callback, early_stopping_callback],
        max_epochs=max_epochs,  # TODO ask ido if we need min epochs
        gpus=1,
        progress_bar_refresh_rate=100,
        log_every_n_steps=100
    )
    return trainer


def update_model_training_params(warmup_steps, total_training_steps, model, i, params):
    # TODO think about a better implemented

    # TODO encapsulate
    model.n_warmup_steps = warmup_steps
    model.n_training_steps = total_training_steps

    if params['weighted_loss'] and ALL_WEIGHTS[i]:
        model.full_weights = ALL_WEIGHTS[i]
        model.weights = True
    else:
        model.weights = False


def train_model(params, data_modules):
    for i, dm in enumerate(data_modules):
        warmup_steps, total_training_steps = calculate_warmup_steps(params, dm)

        if i == 0:
            model = setup_model(warmup_steps, total_training_steps, params['lr'], params['dropout'],
                                params["linear_layer_size"],
                                params['weighted_loss'])
            model.train()
            # model.enable_training()
            # check_encoder_training(model.backbone)
            # check_trainable_layers(model.backbone)

        update_model_training_params(warmup_steps, total_training_steps, model, i, params)

        # TODO ask ido what is the right place for the setup trainer, in the for loop or outside?
        # does it have state?
        trainer = setup_trainer(params['max_epochs'])
        trainer.fit(model, dm)
        # todo : Somewhere above is the saving problem

    trainer.test(model, data_modules[-1])
    return model, trainer


def dump_model_performance_stats(params, model, trainer):
    # TODO change name to be indicative
    base_path = params['train_data']

    # TODO understand what it is
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=True)
    output_dir = "predicted"
    processed_dir = "expected"
    compare_by_file_from_checkpoint(os.path.join(base_path, params['test_data']), output_dir, processed_dir, tokenizer,
                                    100, 5, trainer.checkpoint_callback.best_model_path, params["split_sentence"])

    results = all_stats(output_dir)

    csv_file = OUTPUT_FOLDER / "result_tabel.csv"
    file_exists = csv_file.exists()
    with open(csv_file, "a") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_HEAD))

        if not file_exists:
            writer.writeheader()

        fin = params.copy()
        fin["acc_S"] = model.final_acc_S.item()
        fin["acc_D"] = model.final_acc_D.item()
        fin["acc_N"] = model.final_acc_N.item()
        fin['dec'] = results['dec']
        fin['cha'] = results['cha']
        fin['wor'] = results['wor']
        fin['voc'] = results['voc']
        writer = csv.DictWriter(f, fieldnames=list(CSV_HEAD))
        writer.writerow(fin)


# TODO consider moving it into utils or create model callbacks module

def generate_timestamp() -> str:
    """Generates a timestamp in the format YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class SaveModelCallback(Callback):
    def __init__(self, base_directory: str):
        super().__init__()
        self.base_directory = base_directory

    def save_model_for_huggingface(self, model, epoch: int):
        """Saves the model in Hugging Face format."""
        save_directory = self._create_save_directory(epoch)
        model.save_pretrained(save_directory)

    def _create_save_directory(self, epoch: int) -> Path:
        """Creates a unique directory for saving the model."""
        timestamp = generate_timestamp()
        save_directory = Path(self.base_directory) / f"{timestamp}_epoch_{epoch}"
        save_directory.mkdir(parents=True, exist_ok=True)
        return save_directory

    def on_epoch_end(self, trainer, pl_module):
        """Triggered at the end of each epoch to save the model."""
        epoch = trainer.current_epoch
        self.save_model_for_huggingface(pl_module, epoch)


def run_model(params):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    data_modules = create_data_modules(params)
    model, trainer = train_model(params, data_modules)
    dump_model_performance_stats(params, model, trainer)


try:
    import hydra
    from omegaconf import DictConfig

    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict  # Fallback to regular dict if hydra is unavailable


def parse_config_to_params(cfg: DictConfig) -> Dict:
    # TODO create a dataclass after refactor
    params = {
        "train_data": cfg.dataset.train_path,
        "val_data": cfg.dataset.val_path,
        "test_data": cfg.dataset.test_path,
        "maxlen": cfg.dataset.max_len,
        "minlen": cfg.dataset.min_len,
        "split_sentence": cfg.dataset.split_sentence,
        "lr": cfg.hyper_params.lr,
        "dropout": cfg.hyper_params.dropout,
        "linear_layer_size": cfg.hyper_params.linear_layer_size,
        "train_batch_size": cfg.hyper_params.train_batch_size,
        "val_batch_size": cfg.hyper_params.val_batch_size,
        "max_epochs": cfg.hyper_params.max_epochs,
        "min_epochs": cfg.hyper_params.min_epochs,
        "weighted_loss": cfg.hyper_params.weighted_loss,
        "path": os.getcwd(),
        "version_base": cfg.hydra.version_base
    }
    return params


def main():
    seed_everything(42, workers=True)
    if HYDRA_AVAILABLE:
        @hydra.main(config_path="config", config_name="config")
        def run_with_hydra(cfg: DictConfig):
            print("Hydra is available. Running with slurm's configuration.")
            params = parse_config_to_params(cfg)
            run_model(params)

        run_with_hydra()

    else:
        print("Hydra is not available. Running with default configuration.")
        run_model(DEFAULT_CFG)


if __name__ == "__main__":
    main()
