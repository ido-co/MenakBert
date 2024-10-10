from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule
from sklearn.metrics import ConfusionMatrixDisplay
from torch import Tensor
from torchmetrics import F1Score
from transformers import AutoModel, get_linear_schedule_with_warmup, AutoTokenizer

from consts import N_CLASSES, D_CLASSES, S_CLASSES
from dataset import DAGESH_SIZE, SIN_SIZE, NIQQUD_SIZE, PAD_INDEX


def set_model_training_mode(model: LightningModule, trainable: bool):
    """Set the training mode for all layers in the model.

    Args:
        model: The model whose layers' training mode is to be set.
        trainable (bool): If True, unfreeze all layers (set requires_grad to True).
                          If False, freeze all layers (set requires_grad to False).
    """
    for param in model.parameters():
        param.requires_grad = trainable


def check_encoder_training(model: LightningModule) -> bool:
    all_training = True

    print("Checking encoder layers training mode...")
    for name, module in model.encoder.named_children():
        if not module.training:
            print(f"Layer {name} is NOT in training mode.")
            all_training = False
        else:
            print(f"Layer {name} is in training mode.")

    return all_training


def check_trainable_layers(model: LightningModule) -> bool:
    all_trainable = True

    print("Checking trainable layers...")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Layer {name} is NOT trainable (requires_grad is False).")
            all_trainable = False
        else:
            print(f"Layer {name} is trainable (requires_grad is True).")

    return all_trainable


class NewMenakBert(LightningModule):
    def __init__(self,
                 backbone: str,
                 lr: float,
                 dropout: float,
                 linear_size: int,
                 weights: bool = False,
                 n_training_steps=None,
                 n_warmup_steps=None,
                 trainable: bool = True
                 ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone, cache_dir="models")
        self.backbone.hidden_dropout_prob = dropout

        # Define linear layers
        # TODO can I set 768 as a const? what is it stand for?
        # TODO convert to torch sequential layer
        self.linear_D = nn.Linear(768, DAGESH_SIZE)
        self.linear_S = nn.Linear(768, SIN_SIZE)
        self.linear_N = nn.Linear(768, NIQQUD_SIZE)
        self.linear_up = nn.Linear(768, linear_size)
        self.linear_down = nn.Linear(linear_size, 768)

        # Activation and dropout layers
        self.dropout = nn.Dropout(dropout)
        self.reluLayer1 = nn.ReLU()
        self.reluLayer2 = nn.ReLU()

        self.lr = lr
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps

        # Save hyperparameters
        self.save_hyperparameters()

        # TODO ask / think where to put it
        self.weights = weights

        if self.weights:
            self.set_full_weights()
        else:
            self.full_weights = None

        # Freeze layers if not trainable
        if not trainable:
            self.freeze_layers()

    @staticmethod
    def f1(preds, labels, pad_index=PAD_INDEX):
        """
        Compute F1 score with ignore_index and mdmc_average as 'samplewise'.

        :param preds: Predicted labels (tensor)
        :param labels: True labels (tensor)
        :param pad_index: Index to ignore in the labels (e.g., padding index)
        :return: F1 score (tensor)
        """
        f1_metric = F1Score(ignore_index=pad_index, mdmc_average="samplewise")
        return f1_metric(preds, labels)

    def set_full_weights(self):
        n_weights = torch.tensor([2.4920e-01, 2.6323e-01, 8.8410e-02, 1.0692e-03, 1.2552e-02, 1.4649e-04,
                                  7.2875e-02, 2.8470e-02, 4.6895e-02, 7.4532e-02, 8.5893e-02, 4.8047e-02,
                                  2.6159e-05, 2.8658e-02], device=self.device)
        s_weights = torch.tensor([9.6249e-01, 3.2961e-04, 3.4374e-02, 2.8073e-03], device=self.device)
        d_weights = torch.tensor([0.3914, 0.5126, 0.0961], device=self.device)
        self.full_weights = {'N': n_weights, 'S': s_weights, 'D': d_weights}

    def freeze_layers(self):
        """Freeze all layers to prevent training."""
        # TODO ensure you need both calls
        set_model_training_mode(self, False)
        set_model_training_mode(self.backbone, False)

    def unfreeze_layers(self):
        """Unfreeze all layers to ensure they are trainable."""
        # TODO ensure you need both calls
        set_model_training_mode(self, True)
        set_model_training_mode(self.backbone, True)

    def check_trainable_layers(self):
        """Check if all layers are trainable."""
        model_trainable = all(param.requires_grad for param in self.parameters())
        backbone_trainable = all(param.requires_grad for param in self.backbone.parameters())
        return model_trainable & backbone_trainable

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.backbone(input_ids, attention_mask)['last_hidden_state']
        large = self.linear_up(last_hidden_state)
        drop = self.dropout(large)
        active1 = self.reluLayer1(drop)
        small = self.linear_down(active1)
        active2 = self.reluLayer2(small)
        n = self.linear_N(active2)
        d = self.linear_D(active2)
        s = self.linear_S(active2)
        output = dict(N=n, D=d, S=s)
        return output

    # Helper function to handle cross-entropy loss with optional weights
    @staticmethod
    def weighted_cross_entropy(input, target, pad_index, weight=None):
        return F.cross_entropy(input, target, ignore_index=pad_index, weight=weight)

    def calculate_loss(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        outputs = self(input_ids, attention_mask)

        # Combined logic with conditional weights
        loss_n = self.weighted_cross_entropy(outputs['N'].permute(0, 2, 1), labels['N'].long(), PAD_INDEX,
                                             self.full_weights['N'] if self.weights else None)
        loss_d = self.weighted_cross_entropy(outputs['D'].permute(0, 2, 1), labels['D'].long(), PAD_INDEX,
                                             self.full_weights['D'] if self.weights else None)
        loss_s = self.weighted_cross_entropy(outputs['S'].permute(0, 2, 1), labels['S'].long(), PAD_INDEX,
                                             self.full_weights['S'] if self.weights else None)

        loss = loss_n + loss_d + loss_s
        return loss, outputs

    def training_step(self, batch, batch_idx):
        loss, outputs = self.calculate_loss(batch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": batch["label"]}

    def validation_step(self, batch, batch_idx):
        loss, outputs = self.calculate_loss(batch)
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": batch["label"]}

    def validation_evaluate_layer_metrics(self, layer_key: str, outputs):
        # TODO find a better name
        logits = torch.cat([tmp["predictions"][layer_key] for tmp in outputs])
        pred = torch.argmax(logits, dim=-1)
        labels = torch.cat([tmp["labels"][layer_key] for tmp in outputs])
        pec = self.f1(pred, labels)
        self.log(f'f1_score_{layer_key}', pec, prog_bar=True, logger=True)
        return pec

    def validation_epoch_end(self, output_results):
        pec_S = self.validation_evaluate_layer_metrics("S", output_results)
        pec_D = self.validation_evaluate_layer_metrics("D", output_results)
        pec_N = self.validation_evaluate_layer_metrics("N", output_results)
        return {'f1_score_S': pec_S,
                'f1_score_D': pec_D,
                'f1_score_N': pec_N,
                }

    def test_step(self, batch, batch_idx):
        loss, outputs = self.calculate_loss(batch)
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": batch["label"]}

    def log_confusion_matrix(self, preds: Tensor, labels: Tensor, class_labels: List[str], title: str, tag: str,
                             size: Tuple[int, int] = (12, 7)) -> None:
        """
        Helper function to log confusion matrices.

        Args:
            preds (Tensor): Predicted labels.
            labels (Tensor): True labels.
            class_labels (List[str]): List of class names for display.
            title (str): Title for the confusion matrix plot.
            tag (str): Tag for the logger to group the matrices.
            size (Tuple[int, int]): Size of the figure to display.
        """
        plt.figure(figsize=size)
        ConfusionMatrixDisplay.from_predictions(
            labels.flatten(), preds.flatten(),
            labels=[i for i in range(len(class_labels))],
            display_labels=class_labels,
            normalize="true"
        )
        fig = plt.gcf()
        fig.set_size_inches(1.5 * 18.5, 1.5 * 10.5)
        plt.title(title)
        self.logger.experiment.add_figure(tag, fig)
        plt.close(fig)

    def test_evaluate_layer_metrics(self, layer_key: str, layer_classes: List[str], outputs):
        logits = torch.cat([tmp["predictions"][layer_key] for tmp in outputs])
        pred = torch.argmax(logits, dim=-1)
        labels = torch.cat([tmp["labels"][layer_key] for tmp in outputs])
        f1_res = self.f1(pred, labels)
        labels = labels.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        self.log_confusion_matrix(pred, labels, layer_classes, f"{layer_key} Confusion Matrix",
                                  f"{layer_key} Confusion Matrix")
        return f1_res

    def test_epoch_end(self, output_results):
        self.final_acc_N = self.test_evaluate_layer_metrics("N", N_CLASSES, output_results)
        self.final_acc_D = self.test_evaluate_layer_metrics("D", D_CLASSES, output_results)
        self.final_acc_S = self.test_evaluate_layer_metrics("S", S_CLASSES, output_results)
        return {'train_epoch_f1_precision_S': self.final_acc_S,
                'train_epoch_f1_precision_D': self.final_acc_D,
                'train_epoch_f1_precision_N': self.final_acc_N,
                }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )

    def save_pretrained(self, save_directory: str):
        """Saves the backbone, custom classification heads, and tokenizer."""
        save_path = Path(save_directory)

        # Save the backbone model
        self.backbone.save_pretrained(save_path)


        # Save the state of the custom classification heads
        torch.save(self.linear_D.state_dict(), save_path / 'linear_D.pth')
        torch.save(self.linear_S.state_dict(), save_path / 'linear_S.pth')
        torch.save(self.linear_N.state_dict(), save_path / 'linear_N.pth')
        torch.save(self.linear_up.state_dict(), save_path / 'linear_up.pth')
        torch.save(self.linear_down.state_dict(), save_path / 'linear_down.pth')

    def load_pretrained(self, save_directory: str):
        """Loads the backbone, custom classification heads, and tokenizer."""
        save_path = Path(save_directory)

        # Load the backbone model
        self.backbone.from_pretrained(save_path)

        # Load the state of the custom classification heads
        self.linear_D.load_state_dict(torch.load(save_path / 'linear_D.pth'))
        self.linear_S.load_state_dict(torch.load(save_path / 'linear_S.pth'))
        self.linear_N.load_state_dict(torch.load(save_path / 'linear_N.pth'))
        self.linear_up.load_state_dict(torch.load(save_path / 'linear_up.pth'))
        self.linear_down.load_state_dict(torch.load(save_path / 'linear_down.pth'))
