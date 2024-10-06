import csv
import os
from typing import Dict, List

import gdown
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer

from HebrewDataModule import HebrewDataModule, create_data_modules
from MenakBert import MenakBert, check_encoder_training, check_trainable_layers
from consts import BACKBONE, DEFAULT_CFG, BACKBONE_NAME, ALL_WEIGHTS, BACKBONE_MODEL_LOCAL_URL
from evaluation import compare_by_file_from_checkpoint
from metrics import all_stats
from run_tests import CSV_HEAD

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


def setup_model(data_len, lr, dropout, train_batch_size, max_epochs, min_epochs, linear_size, weights=None):
    # TODO ask ido if it should be part of MenakBert
    download_backbone_model_essentials()

    steps_per_epoch = data_len // train_batch_size
    total_training_steps = steps_per_epoch * max_epochs
    warmup_steps = total_training_steps // 5

    model = MenakBert(backbone=BACKBONE,
                      dropout=dropout,
                      train_batch_size=train_batch_size,
                      lr=lr,
                      linear_size=linear_size,
                      max_epochs=max_epochs,
                      min_epochs=min_epochs,
                      n_warmup_steps=warmup_steps,
                      n_training_steps=total_training_steps,
                      weights=weights)
    return model


def setup_trainer(max_epochs):
    # config training
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="f1_score_N",
        mode="max"
    )

    logger = TensorBoardLogger("lightning_logs", name="nikkud_logs")
    early_stopping_callback = EarlyStopping(monitor='train_loss', patience=5)

    trainer = Trainer(
        logger=logger,
        # auto_lr_find=True,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stopping_callback],
        max_epochs=max_epochs,
        gpus=0, # TODO change to 0 as it was
        progress_bar_refresh_rate=100,
        log_every_n_steps=100
    )
    return trainer

#
# def create_data_modules(params: Dict) -> List[HebrewDataModule]:
#     base_path = params['train_data']
#     # TODO create it dynamically
#     dirs = ['religion', 'pre_modern', 'early_modern', 'modern']
#     testpath = [None, None, None, params['test_data']]
#
#     data_modules = []
#     for i, directory in enumerate(dirs):
#         train_path = os.path.join(base_path, directory)
#
#         dm = HebrewDataModule(
#             train_paths=[train_path],
#             val_path=[params['val_data']],
#             model=BACKBONE,
#             max_seq_length=params['maxlen'],
#             min_seq_length=params['minlen'],
#             train_batch_size=params['train_batch_size'],
#             val_batch_size=params['val_batch_size'],
#             split_sentence=params['split_sentence'],
#             test_paths=[testpath[i]],
#         )
#         dm.setup()
#         data_modules.append(dm)
#     return data_modules


def run_model(params):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    data_modules = create_data_modules(params)
    # TODO change name to be indicative
    base_path = params['train_data']

    for i, dm in enumerate(data_modules):
        if i == 0:
            model = setup_model(len(dm.train_data), params['lr'], params['dropout'], params['train_batch_size'],
                                params['max_epochs'], params['min_epochs'], params["linear_layer_size"],
                                params['weighted_loss'])
            # TODO think what should be the implementation
            model.train()
            model.enable_training()

            check_encoder_training(model.backbone)
            check_trainable_layers(model.backbone)
        else:
            steps_per_epoch = len(dm.train_data) // params['train_batch_size']
            total_training_steps = steps_per_epoch * params['max_epochs']
            warmup_steps = total_training_steps // 5
            model.n_warmup_steps = warmup_steps
            model.n_training_steps = total_training_steps

            if (params['weighted_loss']) and (not (ALL_WEIGHTS[i] is None)):
                model.full_weights = ALL_WEIGHTS[i]
                model.weights = True
            else:
                model.weights = False

        trainer = setup_trainer(params['max_epochs'])
        trainer.fit(model, dm)
        # todo : Somewhere above is the saving problem

    trainer.test(model, data_modules[-1])

    # Mor: copied for previous code under main - when hydra was not available
    # model, dm = setup_model(**params)
    # trainer = setup_trainer(params['max_epochs'])
    # # trainer.tune(model)
    # trainer.fit(model, dm)
    # trainer.test(model, dm)

    with open(os.path.join(base_path, "result_tabel.csv"), "a") as f:
        fin = params.copy()
        fin["acc_S"] = model.final_acc_S.item()
        fin["acc_D"] = model.final_acc_D.item()
        fin["acc_N"] = model.final_acc_N.item()
        writer = csv.DictWriter(f, fieldnames=list(CSV_HEAD))
        writer.writerow(fin)

    # Mor: copied for previous code under main - when hydra was not available

    # model, dm = setup_model(**params)
    # model, dm = setup_model(train_data=train_data,
    #                         val_data=val_data,
    #                         test_data=test_data,
    #                         model=BACKBONE,
    #                         maxlen=MAX_LEN,
    #                         minlen=MIN_LEN,
    #                         lr=LR,
    #                         dropout=DROPOUT,
    #                         train_batch_size=Train_BatchSize,
    #                         val_batch_size=Val_BatchSize,
    #                         max_epochs=MAX_EPOCHS,
    #                         min_epochs=MIN_EPOCHS,
    #                         weighted_loss=True)
    # trainer = setup_trainer(MAX_EPOCHS)
    # # with cProfile.Profile() as pr:
    # #     trainer.fit(model, dm)
    # # stat = pstats.Stats(pr)
    # # stat.dump_stats(filename="run_time.prof")
    # trainer.test(model, dm)
    # CSV_HEAD = [
    #     "train_data",
    #     "val_data",
    #     "test_data",
    #     "model",
    #     "maxlen",
    #     "minlen",
    #     "lr",
    #     "dropout",
    #     "train_batch_size",
    #     "val_batch_size",
    #     "max_epochs",
    #     "min_epochs",
    #     "weighted_loss",
    #     "acc_S",
    #     "acc_D",
    #     "acc_N",
    # ]
    # # with open("result_tabel.csv", "a") as f:
    # #     # writer = csv.writer(f)
    # #     fin = params.copy()
    # #     fin["acc_S"] = model.final_acc_S.item()
    # #     fin["acc_D"] = model.final_acc_D.item()
    # #     fin["acc_N"] = model.final_acc_N.item()
    # #     writer = csv.DictWriter(f, fieldnames=list(CSV_HEAD))
    # #     writer.writerow(fin)

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=True)
    compare_by_file_from_checkpoint(os.path.join(base_path, params['test_data']), r"predicted", r"expected", tokenizer,
                                    100, 5, params["split_sentence"], trainer.checkpoint_callback.best_model_path)
    results = all_stats('predicted')

    with open(os.path.join(base_path, "result_tabel.csv"), "a") as f:
        # writer = csv.writer(f)
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


try:
    import hydra
    from omegaconf import DictConfig

    # TODO make it work properly
    HYDRA_AVAILABLE = False
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
