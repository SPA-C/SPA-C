# -*- coding: utf-8 -*-
"""
Model class

@author: alexis.mergez@inrae.fr
@version: 1.2
@Last modified: 2025/11/12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix, auc, balanced_accuracy_score, average_precision_score
from tqdm import tqdm


class PartialConv2d(nn.Conv2d):
    """
    Official implementation by Guilin Liu (guilinl@nvidia.com). See https://github.com/NVIDIA/partialconv for more !
    """
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output

class ReduceBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReduceBlock, self).__init__()
        self.conv = PartialConv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False, return_mask=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        x, updated_mask = self.conv(x, mask)

        x = self.bn(x)
        x = self.relu(x)
        return x, updated_mask

class PathBlock(nn.Module):
    def __init__(self, channels, stride):
        super(PathBlock, self).__init__()
        self.conv = PartialConv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False, return_mask=True)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        x, updated_mask = self.conv(x, mask)
        x = self.bn(x)
        x = self.relu(x)
        return x, updated_mask
    
class ShortcutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ShortcutBlock, self).__init__()
        self.conv = PartialConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, return_mask=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, mask):
        x, updated_mask = self.conv(x, mask)

        x = self.bn(x)
        return x, updated_mask

class RestoreBlock(nn.Module):
    def __init__(self, channels):
        super(RestoreBlock, self).__init__()
        self.conv = PartialConv2d(channels, channels, kernel_size=1, stride=1, bias=False, return_mask=True)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x, mask):
        x, updated_mask = self.conv(x, mask)

        x = self.bn(x)
        return x, updated_mask

class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cardinality=4, stride=1):
        super(ResNeXtBlock, self).__init__()
        self.cardinality = cardinality
        self.stride = stride
        self.gradcam_mode = False

        # --- Dimensionality reduction (1x1 conv + BatchNorm + ReLU) ---
        self.reduce = ReduceBlock(in_channels, out_channels // cardinality)

        # --- Parallel paths (3x3 convs + BatchNorm + ReLU) ---
        self.paths = nn.ModuleList()
        for _ in range(self.cardinality):
            self.paths.append(PathBlock(out_channels // cardinality, stride))

        # --- Dimensionality restoration (1x1 conv + BatchNorm) ---
        self.restore = RestoreBlock(out_channels)

        # --- Shortcut connection (projection if needed) ---
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = ShortcutBlock(in_channels, out_channels, stride)

    def forward(self, x, mask):
        # Save input for residual connection
        residual = x
        residual_mask = mask

        # Dimensionality reduction
        out, out_mask = self.reduce(x, mask)
        out = out * out_mask # Apply the mask

        # Parallel paths
        path_outs = []
        for path in self.paths:
            path_out, path_mask = path(out, out_mask)
            path_outs.append(path_out)

        out = torch.cat(path_outs, dim=1)  # Concatenation of channels

        # Dimensionality restoration
        out, out_mask = self.restore(out, path_mask)

        # Transforming the shortcut if required
        if isinstance(self.shortcut, ShortcutBlock):
            residual, residual_mask = self.shortcut(residual, residual_mask)

        # Residual connection + activation
        out += residual
        out = out * out_mask

        out = F.relu(out, inplace=True)

        if self.gradcam_mode:
            return out
        return out, out_mask

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias = False):
        super(ConvBlock, self).__init__()
        self.conv = PartialConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, return_mask=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        x, updated_mask = self.conv(x, mask)

        x = self.bn(x)
        x = self.relu(x)
        return x, updated_mask

class encoder(nn.Module):
    def __init__(self, input_shape=(1, 20, 20), latent_width=64):
        super(encoder, self).__init__()

        self.Conv1 = ConvBlock(input_shape[0], latent_width//4, kernel_size=7, stride=2, padding=3, bias=False)

        self.ResBlock1 = ResNeXtBlock(
            in_channels=latent_width//4,
            out_channels=latent_width//2,
            cardinality=4,
            stride=1
        )

        self.ResBlock2 = ResNeXtBlock(
            in_channels=latent_width//2,
            out_channels=latent_width,
            cardinality=4,
            stride=1
        )

        self.ResBlock3 = ResNeXtBlock(
            in_channels=latent_width,
            out_channels=latent_width,
            cardinality=4,
            stride=2
        )

        self.pool = nn.AvgPool2d(5, stride=1)

    def _get_flatten_size(self, input_shape):
        """Compute the size of the flattened layer after convolutions and pooling."""
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)  # Batch size of 1
            x = self.Encoder(x)  # Pass through convolutional layers
            return x.size(1)  # Return the size

    def forward(self, x, return_mask=False):
        mask = (x != -1).float().detach()
        mask_list = {"INIT":mask.detach().clone()}

        x, mask = self.Conv1(x, mask)
        mask_list["Conv1"] = mask.detach().clone()
        x, mask = self.ResBlock1(x, mask)
        mask_list["ResBlock1"] = mask.detach().clone()
        x, mask = self.ResBlock2(x, mask)
        mask_list["ResBlock2"] = mask.detach().clone()
        x, mask = self.ResBlock3(x, mask)
        mask_list["ResBlock3"] = mask.detach().clone()

        x = self.pool(x)
        mask = self.pool(mask)
        mask = mask.expand_as(x)
        mask_list["Pool"] = mask.detach().clone()
        x = x * mask

        x = x.flatten(start_dim=1)

        if return_mask:
            return x, mask_list
        return x

class classifier(nn.Module):
    """Simple MLP"""
    def __init__(self, latent_width=64, num_classes=1):
        super(classifier, self).__init__()

        self.classification_head = nn.Sequential(
            nn.Linear(latent_width, latent_width//2),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(latent_width//2, num_classes)
        )

    def forward(self, x):
        return self.classification_head(x)

class FocalLoss(nn.Module):
    """
    Implementation of the focal loss based on BCE.
    See Lin, T.-Y. et al., 2018. Focal Loss for Dense Object Detection (No. arXiv:1708.02002).
    arXiv. https://doi.org/10.48550/arXiv.1708.02002

    Note:
        Calling with alpha=1 and gamma=1 is equivalent to the classic BCE.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', pos_weight=1, neg_weight=1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Positive frequency
        self.gamma = gamma  # Focus factor
        self.reduction = reduction  # 'mean' ou 'sum'
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, logits, y_true):
        """
        Logits: Logits given by the model
        y_true: True labels
        """

        # Computing BCE without reduction
        bce = F.binary_cross_entropy_with_logits(logits, y_true, reduction='none')

        y_pred = torch.sigmoid(logits)
        # Computing probabilty associated with correct label
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)

        # Focal factor
        focal_weight = (1 - p_t).pow(self.gamma)

        # Weighting loss with class balancing
        if self.alpha is not None:
            alpha_t = (1 - self.alpha) * y_true + self.alpha * (1 - y_true)
        else : # No class balancing
            alpha_t = self.pos_weight * y_true + self.neg_weight * (1 - y_true)

        # Applying focal weighting to loss
        loss = alpha_t * focal_weight * bce

        # Reduction
        return loss.mean() if self.reduction == 'mean' else loss.sum()

class DLScaff:
    def __init__(self, device, name, latent_width=64, input_shape=(1, 20, 20), weights=None):
        self.input_shape = input_shape
        self.latent_width = latent_width
        self.device = device
        self.name = name

        # Assembling the model
        self.encoder = encoder(
            input_shape=input_shape,
            latent_width=latent_width
        )

        self.classification_head = classifier(
            latent_width=self.latent_width,
            num_classes=1
        )

        self.model = nn.Sequential(
            self.encoder,
            self.classification_head
        )

        self.model.to(self.device)

        # Loading weights if provided
        if weights is not None:
            self.load_weights(weights)

    def set_freeze_encoder(self, status=True):
        """
        Set the encoder weights of the classifier to given status
        """
        if status : print("Freezing encoder weights ...")
        else : print("Unfreezing encoder weights ...")

        for param in self.encoder.parameters():
            param.requires_grad = (status == False)

    def training_loop(self, dataLoader, criterion, optimizer, scheduler = None, augmenter=None, description=None, val=False):
        """
        Finetuning loop on a given dataloader

        """
        loss_list = []

        all_preds, all_probs, all_labels = [], [], []

        with tqdm(dataLoader, unit="batch") as tepoch:
            for x, y in tepoch:
                tepoch.set_description(description)

                x = x.to(self.device)
                y = y.to(self.device)

                if augmenter is not None and not val:
                    z = self.model(augmenter(x))
                else:
                    z = self.model(x)

                loss = criterion(z.squeeze(), y)
                loss_list.append(loss.item())

                if not val:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                tepoch.set_postfix(loss=loss.item(), avg_loss=np.mean(loss_list))

                probs = torch.sigmoid(z)
                predictions = (probs.detach() >= 0.5).long()

                all_probs.append(probs.detach().cpu())
                all_preds.append(predictions.cpu())
                all_labels.append(y.cpu())

        if scheduler is not None:
            scheduler.step()

        # Concatenating results
        all_preds = torch.cat(all_preds).numpy()
        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # Computing train accuracy, etc...
        accuracy = balanced_accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds,
                                    average='weighted')  # Weighted Precision based on support for each class
        recall = recall_score(all_labels, all_preds,
                              average='weighted')  # Weighted Recall based on support for each class
        wf1 = f1_score(all_labels, all_preds, average='weighted',
                       labels=[0, 1])  # Weighted F1 based on support for each class
        mf1 = f1_score(all_labels, all_preds, average='macro', labels=[0, 1])
        roc_auc = roc_auc_score(all_labels, all_probs, average='weighted')  # Weighted AUC
        precision4curve, recall4curve, _ = precision_recall_curve(all_labels, all_preds)
        aupr = average_precision_score(all_labels, all_preds, average="weighted")  # Weighted average precision
        cm = confusion_matrix(all_labels, all_preds)
        tn_, fp_, fn_, tp_ = cm.ravel()
        NPV = tn_ / (fn_ + tn_)

        avg_loss = np.mean(loss_list)

        return avg_loss, accuracy, precision, recall, wf1, mf1, roc_auc, aupr, NPV

    def training(
            self,
            dataLoader,
            savedir,
            val_dataLoader=None,
            augmenter=None,
            nepochs=20,
            freeze_encoder=False,
            FocalLoss_alpha=None,
            FocalLoss_gamma=1,
            FocalLoss_pos_weight=1,
            FocalLoss_neg_weight=1,
            add_scheduler=False,
            lr=0.001,
            start_epoch=0
        ):

        # Freezing encoder weights if required
        self.set_freeze_encoder(status = freeze_encoder)


        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )

        if add_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nepochs)
        else : scheduler = None

        log_file = os.path.join(savedir, f"{self.name}.Classifier.log")
        with open(log_file, "w") as handle:
            handle.write(
                "Epoch\tTrain loss\tTrain accuracy\tTrain precision\tTrain recall\tTrain weighted F1\tTrain macro F1\tTrain ROC AUC\tTrain weighted AP\tTrain NPV" +\
                    "\tVal loss\tVal accuracy\tVal precision\tVal recall\tVal weighted F1\tVal macro F1\tVal ROC AUC\tVal weighted AP\tVal NPV"*(val_dataLoader is not None) +\
                    "\n"
            )

        criterion = FocalLoss(
            alpha=FocalLoss_alpha,
            gamma=FocalLoss_gamma,
            pos_weight=FocalLoss_pos_weight,
            neg_weight=FocalLoss_neg_weight
        )  # Focal loss (= BCE with default values)

        for epoch in range(start_epoch, start_epoch + nepochs):
            # Training step
            self.model.train()
            train_avg_loss, accuracy, precision, recall, wf1, mf1, roc_auc, aupr, NPV = self.training_loop(
                dataLoader=dataLoader,
                augmenter=augmenter,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                description=f"Train | Epoch {epoch:>02}",
                val=False
            )
            print(
                f"Train | Epoch: {epoch:>02}, average loss: {train_avg_loss:.5f}, accuracy: {accuracy:.5f}, wF1: {wf1:.5f}, mF1: {mf1:.5f}, AUC: {roc_auc:.5f}, AUPR: {aupr:.5f}, NPV: {NPV:.5f}"
            )
            if scheduler is not None:
                print(
                    f"Train | Epoch: {epoch:>02} | Last LR:{scheduler.get_last_lr()[0]}"
                )

            # Validation step
            if val_dataLoader is not None:
                self.model.eval()

                with torch.no_grad():
                    val_avg_loss, v_accuracy, v_precision, v_recall, v_wf1, v_mf1, v_roc_auc, v_aupr, v_NPV = self.training_loop(
                        dataLoader=val_dataLoader,
                        augmenter=augmenter,
                        criterion=criterion,
                        optimizer=optimizer,
                        description=f"Val | Epoch {epoch:>02}",
                        val=True
                    )
                print(
                    f"Val | Epoch: {epoch:>02}, average loss: {val_avg_loss:.5f}, accuracy: {v_accuracy:.5f}, wF1: {v_wf1:.5f}, mF1: {v_mf1:.5f}, AUC: {v_roc_auc:.5f}, AUPR: {v_aupr:.5f}, NPV: {v_NPV:.5f}"
                )

            self.save_weights(
                savedir = os.path.join(savedir, f"{self.name}.Classifier.E{epoch:>02}.pth")
            )

            with open(log_file, "a") as handle:
                handle.write(
                    f"{epoch:>02}\t{train_avg_loss:.5f}\t{accuracy:.5f}\t{precision:.5f}\t{recall:.5f}\t{wf1:.5f}\t{mf1:.5f}\t{roc_auc:.5f}\t{aupr:.5f}\t{NPV:.5f}" +\
                        f"\t{val_avg_loss:.5f}\t{v_accuracy:.5f}\t{v_precision:.5f}\t{v_recall:.5f}\t{v_wf1:.5f}\t{v_mf1:.5f}\t{v_roc_auc:.5f}\t{v_aupr:.5f}\t{v_NPV:.5f}"*(val_dataLoader is not None) +\
                        "\n"
                )

    def save_weights(self, savedir, debug=True):
        torch.save(self.model.state_dict(), savedir)
        if debug: print("Classifier weights saved !")

    def load_weights(self, dir, debug=True):
        self.model.load_state_dict(torch.load(dir, weights_only=True, map_location=torch.device(self.device)))
        if debug: print("Weights loaded !")

    def predict(self, dataLoader, savedir=None, names=None):
        """
        Use the model to predict on the provided dataLoader, on the provided device.
        """
        self.model.eval()

        # Creating list to store predictions, raw outputs and labels
        all_probs = []
        all_labels = []
        all_names = []

        with torch.no_grad():
            with tqdm(dataLoader, unit="batch") as tepoch:
                for x, y in tepoch:
                    tepoch.set_description(f"Predicting on test")
                    x = x.to(self.device)
                    outputs = self.model(x)

                    probs = torch.sigmoid(outputs)
                    all_probs.append(probs.clone().cpu())
                    all_labels.append(y.clone().cpu())

        # Concatenating predictions, raw outputs and labels
        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # Exporting predictions to JSON
        if savedir is not None and names is not None:
            prob_dict = {names[i]: float(all_probs[i][0]) for i in range(len(names))}

            with open(savedir, "w") as handle:
                json.dump(
                    prob_dict,
                    handle,
                    indent=2
                )

        return all_probs, all_labels

    def __get_text_color__(self, background_color):
        # Compute luminance and decide the color of the text.
        r, g, b = background_color[:3]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return 'white' if luminance < 0.5 else 'black'

    def eval(self, dataLoader, threshold=.5, savedir=None, names=None, return_data=False, title=None):
        """
        Compute several metrics on the model using he provided dataset.
        """
        # Predicting
        all_probs, all_labels = self.predict(
            dataLoader=dataLoader,
            savedir=savedir,
            names=names
        )

        fs = 5
        plt.rcParams.update({'font.size': fs, 'legend.fontsize': fs, 'figure.labelsize': fs})

        fig, axs = plt.subplots(2, 2, figsize=(3, 3), dpi=600)

        # ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = roc_auc_score(all_labels, all_probs, average='weighted')  # Weighted AUC
        axs[0, 0].plot(fpr, tpr, label=f"AUC-ROC = {roc_auc:.2f}")
        axs[0, 0].set_xlabel("False Positive Rate")
        axs[0, 0].set_ylabel("True Positive Rate")
        axs[0, 0].set_title("ROC Curve")
        axs[0, 0].legend()
        axs[0, 0].spines["top"].set_visible(False)
        axs[0, 0].spines["right"].set_visible(False)

        """
        # Youden's J index (best threshold)
        #best_thres = thresholds[np.argmax(np.array(tpr) - np.array(fpr))]

        # F1 macro best threshold selector
        best_thres = thresholds[np.argmax(
            [f1_score(
                all_labels, 
                [ int(k[0] >= thres) for k in all_probs],
                average='macro',
                labels=[0, 1]
            ) for thres in thresholds]
        )]
        threshold = best_thres
        """

        all_preds_best = [int(k[0] >= threshold) for k in all_probs]

        # Metrics computation
        accuracy = balanced_accuracy_score(all_labels, all_preds_best)
        precision = precision_score(all_labels, all_preds_best,
                                    average='weighted')  # Weighted Precision based on support for each class
        recall = recall_score(all_labels, all_preds_best,
                              average='weighted')  # Weighted Recall based on support for each class
        f1 = f1_score(all_labels, all_preds_best, average='weighted',
                      labels=[0, 1])  # Weighted F1 based on support for each class
        mf1 = f1_score(all_labels, all_preds_best, average='macro',
                       labels=[0, 1])  # Weighted F1 based on support for each class
        precision4curve, recall4curve, _ = precision_recall_curve(all_labels, all_probs)
        aupr = average_precision_score(all_labels, all_probs, average="weighted")  # Weighted average precision
        cm = confusion_matrix(all_labels, all_preds_best)
        tn_, fp_, fn_, tp_ = cm.ravel()
        NPV = tn_ / (fn_ + tn_)

        print(
            f"\n\tBalanced Accuracy: {accuracy:.4f}\n\tWeighted precision: {precision:.4f}\n\tWeighted recall: {recall:.4f}\n\tWeighted F1: {f1:.4f}\n\tMacro F1: {mf1:.4f}\n\tWeighted AUROC: {roc_auc:.4f}\n\tWeighted AP: {aupr:.4f}\n\tNegative predictive value: {NPV:.4f}\n")

        # AUPR curve
        axs[1, 0].plot(recall4curve, precision4curve, label=f"AUPR = {aupr:.4f}")
        axs[1, 0].set_xlabel("Recall")
        axs[1, 0].set_ylabel("Precision")
        axs[1, 0].set_title("Precision-Recall Curve")
        axs[1, 0].legend()
        axs[1, 0].spines["top"].set_visible(False)
        axs[1, 0].spines["right"].set_visible(False)

        # Confusion matrix /OLD/
        cm_sum = cm.sum(axis=1, keepdims=True)
        conf_mat_display = np.divide(cm, cm_sum, where=cm_sum != 0) * 100

        #
        # sns.heatmap(cm, annot=False, cmap="Blues", ax=axs[0, 1])
        # sns.heatmap(cm, annot=True, fmt="d", annot_kws={'va': 'bottom', 'fontweight': 'bold'}, cmap="Blues",
        #             ax=axs[0, 1], cbar=False)
        # sns.heatmap(conf_mat_str, annot=True, annot_kws={'va': 'top'}, cmap="Blues", ax=axs[0, 1],
        #             cbar=False)

        heatmap = sns.heatmap(
            cm,
            annot=False,
            cmap="Blues",
            ax=axs[0, 1],
            cbar=False,
        )
        axs[0, 1].tick_params(axis='y', labelrotation=0)
        for spine in axs[0, 1].spines.values():
            spine.set_visible(True)

        facecolors = heatmap.collections[0].get_facecolors()

        ## Adding personnalized annotation
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):

                cell_color = facecolors[i * cm.shape[1] + j]
                text_color = self.__get_text_color__(cell_color)

                axs[0, 1].text(
                    j + 0.5,
                    i + 0.6,
                    f"{conf_mat_display[i, j]:.2f}%",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=fs,
                )
                axs[0, 1].text(
                    j + 0.5,
                    i + 0.4,
                    f"{cm[i, j]}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=fs,
                    fontweight="bold",
                )

        axs[0, 1].set_xlabel("Predicted label")
        axs[0, 1].set_ylabel("True label")
        axs[0, 1].set_title("Confusion Matrix")

        # Scores
        data = [
            ["Threshold", f"{threshold:.3f}"],
            ["NPV", f"{NPV:.3f}"],
            ["Weighted AP", f"{aupr:.3f}"],
            ["Weighted AUROC", f"{roc_auc:.3f}"],
            ["Macro F1", f"{mf1:.3f}"],
            ["Weighted F1", f"{f1:.3f}"],
            ["Weighted recall", f"{recall:.3f}"],
            ["Weighted precision", f"{precision:.3f}"],
            ["Balanced Accuracy", f"{accuracy:.3f}"]
        ]
        axs[1, 1].axis("off")
        table = axs[1, 1].table(
            cellText=data,
            colLabels=["", ""],
            loc='center',
            cellLoc='left',
            colWidths=[0.75, 0.25],

        )

        for (row, col), cell in table.get_celld().items():
            if col == 1 and row > 0:
                cell.set_edgecolor('black')
                cell.set_linewidth(.5)
                cell.visible_edges="L"
            else:
                cell.set_edgecolor('none')

        table.auto_set_font_size(False)
        table.set_fontsize(fs)

        """
        x_pos = 0
        axs[1, 1].axis([0, 10, 0, 10])
        axs[1, 1].text(x_pos, 1, f"Balanced Accuracy: {accuracy:.3f}")
        axs[1, 1].text(x_pos, 2, f"Weighted precision: {precision:.3f}")
        axs[1, 1].text(x_pos, 3, f"Weighted recall: {recall:.3f}")
        axs[1, 1].text(x_pos, 4, f"Weighted F1: {f1:.3f}")
        axs[1, 1].text(x_pos, 5, f"Macro F1: {mf1:.3f}")
        axs[1, 1].text(x_pos, 6, f"Weighted AUROC: {roc_auc:.3f}")
        axs[1, 1].text(x_pos, 7, f"Weighted AP: {aupr:.3f}")
        axs[1, 1].text(x_pos, 8, f"Negative predictive value: {NPV:.3f}")
        axs[1, 1].text(x_pos, 9, f"Threshold: {threshold:.3f}", fontweight="bold")
        """

        letters = ["A", "B", "C", "D"]
        for idx, ax in enumerate(axs.flat):
            ax.annotate(
                letters[idx],
                xy=(-0.2, 1.2),
                xycoords="axes fraction",
                fontsize=fs+1,
                fontweight="bold",
                color="black",
                va="top",
                ha="left",
            )

        if title is not None: fig.suptitle(title)
        plt.tight_layout()

        if savedir is not None: plt.savefig(savedir)

        plt.show()

        if return_data:
            return all_preds_best, all_probs, all_labels