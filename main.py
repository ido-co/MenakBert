import csv
import os

import gdown
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer

from HebrewDataModule import HebrewDataModule
from MenakBert import MenakBert
from consts import BACKBONE, TRAIN_PATH, VAL_PATH, TEST_PATH, MAX_LEN, MIN_LEN, LR, DROPOUT, Train_BatchSize, \
    Val_BatchSize, MAX_EPOCHS, MIN_EPOCHS, BACKBONE_NAME, LINEAR_LAYER_SIZE, CSV_HEAD
from evaluation import compare_by_file_from_checkpoint
from metrics import all_stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL_LINK = "https://drive.google.com/drive/folders/1K78B5SM8FjBc_5r-UWTwoj1x105xpksK?usp=sharing"
MODEL = "tavbert"


def setup_dm(train_data, val_data, test_data, model, maxlen, minlen, train_batch_size, val_batch_size):
    dm = HebrewDataModule(
        train_paths=train_data,
        val_path=val_data,
        test_paths=test_data,
        model=model,
        max_seq_length=maxlen,
        min_seq_length=minlen,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size)
    dm.setup()
    return dm


def calc_weights(train_data):
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


def setup_model(data_len, lr, dropout, train_batch_size, max_epochs, min_epochs, linear_size, weights=None):
    # init data module
    if not os.path.exists(BACKBONE_NAME):
        os.mkdir(BACKBONE_NAME)
        gdown.download_folder("https://drive.google.com/drive/folders/1K78B5SM8FjBc_5r-UWTwoj1x105xpksK?usp=sharing",
                              output=BACKBONE_NAME)

    steps_per_epoch = data_len // train_batch_size
    total_training_steps = steps_per_epoch * max_epochs
    warmup_steps = total_training_steps // 5

    # init module
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
        gpus=1,
        progress_bar_refresh_rate=100,
        log_every_n_steps=100
    )
    return trainer


# @hydra.main(config_path="config", config_name="config")
def runModel(cfg: DictConfig):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    global MODEL
    MODEL = os.path.join(f"{cfg.base_path}", "tavbert")
    print(OmegaConf.to_yaml(cfg))

    print(os.getcwd())
    # all model params from the CFG
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

    base_path = cfg.base_path
    dirs = ['religion', 'pre_modern', 'early_modern', 'modern']

    testpath = [None, None, None,
                os.path.join(cfg.base_path, params['test_data'])]  # todo add test path on last element
    # dirs = ['train', 'validation', 'test']
    data_modules = []
    for i, directory in enumerate(dirs):
        train_path = os.path.join(params['train_data'], directory)
        # dm = setup_dm([train_path],
        #               [params['val_data']],
        #               testpath[i],
        #               MODEL,
        #               params['maxlen'],
        #               params['minlen'],
        #               params['train_batch_size'],
        #               params['val_batch_size']
        #               )
        dm = HebrewDataModule(
            train_paths=[os.path.join(base_path, train_path)],
            val_path=[os.path.join(base_path, params['val_data'])],
            test_paths=[testpath[i]],
            model=MODEL,
            max_seq_length=params['maxlen'],
            min_seq_length=params['minlen'],
            train_batch_size=params['train_batch_size'],
            val_batch_size=params['val_batch_size'],
            split_sentence=params['split_sentence']
        )
        dm.setup()
        data_modules.append(dm)
        # weights for all 4 dm
    all_weights = \
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

    for i, dm in enumerate(data_modules):
        if i == 0:
            model = setup_model(len(dm.train_data), params['lr'], params['dropout'], params['train_batch_size'],
                                params['max_epochs'], params['min_epochs'], linear_size=params["linear_layer_size"],
                                weights=params['weighted_loss'])
        else:
            steps_per_epoch = len(dm.train_data) // params['train_batch_size']
            total_training_steps = steps_per_epoch * params['max_epochs']
            warmup_steps = total_training_steps // 5
            model.n_warmup_steps = warmup_steps
            model.n_training_steps = total_training_steps
            if (params['weighted_loss']) and (not (all_weights[i] is None)):
                model.full_weights = all_weights[i]
                model.weights = True
            else:
                model.weights = False

        trainer = setup_trainer(params['max_epochs'])
        trainer.fit(model, dm)
        # todo : Somewhere above is the saving problem

    trainer.test(model, data_modules[-1])

    with open(os.path.join(base_path, "result_tabel.csv"), "a") as f:
        # writer = csv.writer(f)
        fin = params.copy()
        fin["acc_S"] = model.final_acc_S.item()
        fin["acc_D"] = model.final_acc_D.item()
        fin["acc_N"] = model.final_acc_N.item()
        writer = csv.DictWriter(f, fieldnames=list(CSV_HEAD))
        writer.writerow(fin)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    compare_by_file_from_checkpoint(os.path.join(base_path, testpath[-1]), r"predicted", r"expected", tokenizer,
                                    100, 5, trainer.checkpoint_callback.best_model_path, params["split_sentence"])
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


if __name__ == '__main__':
    seed_everything(42, workers=True)
    # runModel()
    params = {
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
        "weighted_loss": True
    }

    base_path = params['train_data']
    dirs = ['religion', 'pre_modern', 'early_modern', 'modern']
    testpath = [None, None, None, params["test_data"]]
    # dirs = ['train', 'val', 'test']
    data_modules = []
    for i, directory in enumerate(dirs):
        # train_path = base_path + "/" + directory
        train_path = os.path.join(base_path, directory)
        # dm = setup_dm([train_path], [params['val_data']], MODEL, params['maxlen'],
        #               params['minlen'], params['train_batch_size'], params['val_batch_size'])

        dm = HebrewDataModule(
            train_paths=[train_path],
            val_path=[params['val_data']],
            model=BACKBONE,
            max_seq_length=params['maxlen'],
            min_seq_length=params['minlen'],
            train_batch_size=params['train_batch_size'],
            val_batch_size=params['val_batch_size'],
            split_sentence=False,
            test_paths=[testpath[i]]
        )
        dm.setup()
        data_modules.append(dm)

    for i, dm in enumerate(data_modules):
        if i == 0:
            model = setup_model(len(dm.train_data), params['lr'], params['dropout'], params['train_batch_size'],
                                params['max_epochs'], params['min_epochs'], params['linear_layer_size'],
                                weights=params['weighted_loss'])
        else:
            steps_per_epoch = len(dm.train_data) // params['train_batch_size']
            total_training_steps = steps_per_epoch * params['max_epochs']
            warmup_steps = total_training_steps // 5
            model.n_warmup_steps = warmup_steps
            model.n_training_steps = total_training_steps
        trainer = setup_trainer(params['max_epochs'])
        trainer.fit(model, dm)
    trainer.test(model, data_modules[-1])

    model, dm = setup_model(**params)
    trainer = setup_trainer(params['max_epochs'])
    # trainer.tune(model)
    trainer.fit(model, dm)
    trainer.test(model, dm)
    with open("result_tabel.csv", "a") as f:
        # writer = csv.writer(f)
        fin = params.copy()
        fin["acc_S"] = model.final_acc_S
        fin["acc_D"] = model.final_acc_D
        fin["acc_N"] = model.final_acc_N
        writer = csv.DictWriter(f, fieldnames=list(CSV_HEAD))
        writer.writerow(params)

    model, dm = setup_model(**params)
    model, dm = setup_model(train_data=train_data,
                            val_data=val_data,
                            test_data=test_data,
                            model=MODEL,
                            maxlen=MAX_LEN,
                            minlen=MIN_LEN,
                            lr=LR,
                            dropout=DROPOUT,
                            train_batch_size=Train_BatchSize,
                            val_batch_size=Val_BatchSize,
                            max_epochs=MAX_EPOCHS,
                            min_epochs=MIN_EPOCHS,
                            weighted_loss=True)
    trainer = setup_trainer(MAX_EPOCHS)
    # with cProfile.Profile() as pr:
    #     trainer.fit(model, dm)
    # stat = pstats.Stats(pr)
    # stat.dump_stats(filename="run_time.prof")
    trainer.test(model, dm)

    ]
    # with open("result_tabel.csv", "a") as f:
    #     # writer = csv.writer(f)
    #     fin = params.copy()
    #     fin["acc_S"] = model.final_acc_S.item()
    #     fin["acc_D"] = model.final_acc_D.item()
    #     fin["acc_N"] = model.final_acc_N.item()
    #     writer = csv.DictWriter(f, fieldnames=list(CSV_HEAD))
    #     writer.writerow(fin)
