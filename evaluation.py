import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from MenakBert import MenakBert
from consts import BACKBONE
from dataset import textDataset
from metrics import format_output_y1, all_stats


def compare_by_file_from_checkpoint(
        test_path: str,
        output_dir: str,
        processed_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_len: int,
        min_len: int,
        checkpoint_path: str,
        split_sentence: bool,
        logger: logging.Logger = None) -> None:
    """Compares file predictions from a model checkpoint."""
    trained_model = MenakBert.load_from_checkpoint(checkpoint_path)
    compare_by_file_from_model(
        test_path=Path(test_path),
        output_dir=Path(output_dir),
        processed_dir=Path(processed_dir),
        tokenizer=tokenizer,
        max_len=max_len,
        min_len=min_len,
        trained_model=trained_model,
        split_sentence=split_sentence,
        logger=logger
    )


def compare_by_file_from_model(
        test_path: Path,
        output_dir: Path,
        processed_dir: Path,
        tokenizer: PreTrainedTokenizer,
        max_len: int,
        min_len: int,
        trained_model: torch.nn.Module,
        split_sentence: bool,
        logger: logging.Logger = None) -> None:
    """Compares file predictions using a trained model."""
    trained_model.freeze()

    # Ensure output and processed directories exist
    inner_out = output_dir / "inner"
    inner_pro = processed_dir / "inner"
    create_directories([output_dir, inner_out, processed_dir, inner_pro])

    # Iterate over the files in test_path and process them
    for counter, (root, _, files) in enumerate(os.walk(test_path), 1):
        for name in files:
            process_single_file(
                root=root,
                file_name=name,
                counter=counter,
                inner_out=inner_out,
                inner_pro=inner_pro,
                max_len=max_len,
                min_len=min_len,
                tokenizer=tokenizer,
                split_sentence=split_sentence,
                trained_model=trained_model,
                logger=logger
            )


def process_single_file(
        root: str,
        file_name: str,
        counter: int,
        inner_out: Path,
        inner_pro: Path,
        max_len: int,
        min_len: int,
        tokenizer: PreTrainedTokenizer,
        split_sentence: bool,
        trained_model: torch.nn.Module,
        logger: logging.Logger = None) -> None:
    """Processes a single file and generates predictions."""
    new_name = f"{counter}_{file_name}"
    curr_in = Path(root) / file_name
    curr_out = inner_out / new_name
    curr_pro = inner_pro / new_name

    val_dataset = textDataset([str(curr_in)], max_len, min_len, tokenizer, split_sentence)
    loader = DataLoader(val_dataset, batch_size=32, num_workers=8)

    try:
        with curr_out.open('a', encoding='utf8') as f_out, curr_pro.open('a', encoding='utf8') as f_pro:
            process_batches(loader, trained_model, tokenizer, f_out, f_pro, logger)
    except Exception as e:
        if logger:
            logger.error(f"Error processing {file_name}: {e}")
        else:
            print(f"Error processing {file_name}: {e}")


def process_batches(
        loader: DataLoader,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        f_out,
        f_pro,
        logger: logging.Logger = None) -> None:
    """Processes data in batches and writes the predictions and ground truths to files."""
    for batch in loader:
        preds = model(batch['input_ids'], batch['attention_mask'])
        preds['N'] = torch.argmax(preds['N'], dim=-1)
        preds['D'] = torch.argmax(preds['D'], dim=-1)
        preds['S'] = torch.argmax(preds['S'], dim=-1)

        for sent in range(len(preds['N'])):
            # Writing predicted output
            line_out = format_output_y1(
                batch['input_ids'][sent], preds['N'][sent], preds['D'][sent], preds['S'][sent], tokenizer)
            f_out.write(f'{line_out}\n')

            # Writing processed (ground truth) output
            line_pro = format_output_y1(
                batch['input_ids'][sent], batch['label']['N'][sent], batch['label']['D'][sent],
                batch['label']['S'][sent], tokenizer)
            f_pro.write(f'{line_pro}\n')

    if logger:
        logger.info("Batch processing complete.")


def create_compare_file(
        test_path: str,
        target_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_len: int,
        min_len: int,
        split_sentence: bool,
        checkpoint_path: str) -> None:
    """Generates a comparison file using a model checkpoint."""
    trained_model = MenakBert.load_from_checkpoint(checkpoint_path)
    trained_model.freeze()

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(test_path):
        for name in files:
            curr_in = Path(root) / name
            curr_out = target_dir_path / name
            val_dataset = textDataset([str(curr_in)], max_len, min_len, tokenizer, split_sentence)

            with curr_out.open('a', encoding='utf8') as f:
                for sent in range(len(val_dataset)):
                    line = format_output_y1(
                        val_dataset[sent]['input_ids'],
                        val_dataset[sent]['label']['N'],
                        val_dataset[sent]['label']['D'],
                        val_dataset[sent]['label']['S'],
                        tokenizer
                    )
                    f.write(f'{line}\n')


def create_directories(paths: list[Path]) -> None:
    """Creates directories if they don't already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def validate_paths(paths: list[Path]) -> None:
    """Checks if paths are valid and accessible."""
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")


if __name__ == '__main__':
    from transformers import AutoTokenizer
    import logging

    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=True)
    compare_by_file_from_checkpoint(
        test_path=r"hebrew_diacritized_testing2/data/test",
        output_dir=r"predicted",
        processed_dir=r"expected",
        tokenizer=tokenizer,
        max_len=100,
        min_len=5,
        checkpoint_path="/Users/mormatalon/Documents/LevelUp/MenakBert/checkpoints/best-checkpoint-v3.ckpt",
        split_sentence=False,
        logger=logger
    )
    results = all_stats("predicted")
