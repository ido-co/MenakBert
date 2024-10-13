import csv

import hydra
from omegaconf import DictConfig
import os

from consts import CSV_HEAD


@hydra.main(config_path="config", config_name="config")
def run_test(cfg: DictConfig):
    if not os.path.exists(f"{cfg.base_path}/result_tabel.csv"):
        with open(f"{cfg.base_path}/result_tabel.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEAD)
    create_slurm(cfg)


def create_slurm(cfg: DictConfig):
    f = open("menakbert_job.slurm", "w")
    f.write("#! /bin/sh\n")
    f.write("#SBATCH --job-name=menakbert\n")
    f.write("#SBATCH --output=menakbert.out\n")
    f.write("#SBATCH --error=menakbert.err\n")
    f.write("#SBATCH --partition=studentbatch\n")
    f.write(f"#SBATCH --time={60 * cfg.hyper_params.max_epochs}\n")
    f.write("#SBATCH --nodes=1\n")
    f.write("#SBATCH --ntasks=1\n")
    f.write("#SBATCH --gpus=1\n")
    f.write(
        f"python {cfg.base_path}/main.py base_path={cfg.base_path} hyper_params.lr={cfg.hyper_params.lr} "
        f"hyper_params.dropout={cfg.hyper_params.dropout} hyper_params.max_epochs={cfg.hyper_params.max_epochs} "
        f"hyper_params.train_batch_size={cfg.hyper_params.train_batch_size} "
        f"hyper_params.linear_layer_size={cfg.hyper_params.linear_layer_size} "
        f"dataset.max_len={cfg.dataset.max_len} "
        f"hyper_params.weighted_loss={cfg.hyper_params.weighted_loss} "
        f"dataset.split_sentence={cfg.dataset.split_sentence}")
    f.close()
    os.system("sbatch menakbert_job.slurm")


if __name__ == '__main__':
    run_test()
