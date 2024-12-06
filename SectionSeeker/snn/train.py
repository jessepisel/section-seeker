from torch import nn
import torch
import numpy as np
from typing import Sequence, Callable, Union
from snn.losses import ContrastiveLoss, TripletLoss
from functools import partial
from snn.utils import detect_device


def train_classification(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.cuda.device = None,
    optimizer: torch.optim.Optimizer = None,
    criterion: Callable = None,
    epoch: int = 0,
    dry_run: bool = False,
    log_interval: int = 10,
):
    """
    Train model with binary target for classification task with Binary Cross Entropy (or similar metric based on binary targets)

    Requires data loader to output sequences of [img_1, img_2, target],
    where target is 0 for objects coming from different classess and 1 for objects coming from the same class
    """
    if device is None:
        device = detect_device()

    model.to(device).train()
    total_loss = 0

    if criterion is None:
        criterion = nn.BCELoss()

    for batch_idx, ((images_1, images_2), targets) in enumerate(train_loader):
        images_1, images_2, targets = (
            images_1.to(device),
            images_2.to(device),
            targets.to(device),
        )
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(images_1),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if dry_run:
                break
        total_loss /= batch_idx + 1
    return total_loss


def train_contrastive(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.cuda.device = None,
    optimizer: torch.optim.Optimizer = None,
    criterion: Callable = None,
    epoch: int = 0,
    dry_run: bool = False,
    log_interval: int = 10,
):
    """
    Train model with binary target for classification task with Contrastive Loss

    Requires data loader to output sequences of [img_1, img_2, target],
    where target is 0 for objects coming from different classess and 1 for objects coming from the same class
    """
    if device is None:
        device = detect_device()

    model.to(device).train()
    total_loss = 0

    if criterion is None:
        criterion = ContrastiveLoss()

    for batch_idx, ((images_1, images_2), targets) in enumerate(train_loader):
        images_1, images_2, targets = (
            images_1.to(device),
            images_2.to(device),
            targets.to(device),
        )
        optimizer.zero_grad()
        outputs_positive = model.get_embedding(images_1).squeeze()
        outputs_negative = model.get_embedding(images_2).squeeze()
        loss = criterion(outputs_positive, outputs_negative, targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(images_1),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if dry_run:
                break
        total_loss /= batch_idx + 1
    return total_loss


def train_triplet(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.cuda.device = None,
    optimizer: torch.optim.Optimizer = None,
    criterion: Callable = None,
    epoch: int = 0,
    dry_run: bool = False,
    log_interval: int = 10,
):
    """
    Train embedding model with Triplet Margin Loss (or any other divergence based loss) for one epoch

    Requires data loader to output sequences of 3 images: [anchor_img, positive_img, negative_img],
    where anchor image and positive image come from the same class, while nagative image
    comes from the different class
    """
    if device is None:
        device = detect_device()

    model.to(device).train()

    if criterion is None:
        criterion = nn.TripletMarginLoss()
    total_loss = 0

    for batch_idx, ((anchor, positive, negative), _) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs_positive = model.get_embedding(positive).squeeze()
        outputs_negative = model.get_embedding(negative).squeeze()
        outputs_anchor = model.get_embedding(anchor).squeeze()
        loss = criterion(outputs_anchor, outputs_positive, outputs_negative)
        total_loss += loss.item()
        loss.backward()
        optimizer.step
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(images_1),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if dry_run:
                break
        total_loss /= batch_idx + 1
    return total_loss


class Trainer:
    def __init__(
        self,
        loss: Union[Callable, str],
        optimizer,
        cuda: bool,
        log_interval: int = 100,
        metrics: Sequence = None,
    ):
        if type(loss) is str:
            if loss == "contrastive":
                loss = ContrastiveLoss()
            elif loss == "bce":
                loss = nn.BCELoss()
            elif loss == "triplet_margin":
                loss = nn.TripletMarginLoss()
            elif loss == "triplet":
                loss = TripletLoss()
            else:
                raise ValueError("Unknown loss specified")

        self.loss = loss
        self.optimizer = optimizer
        self.device = torch.device("cuda") if cuda else torch.device("cpu")
        self.metrics = metrics
        self.log_interval = log_interval

        # prepare training routine
        if isinstance(self.loss, nn.BCELoss):
            self.training_epoch = partial(
                train_classification,
                device=self.device,
                optimizer=self.optimizer,
                criterion=self.loss,
                log_interval=self.log_interval,
            )

        elif isinstance(self.loss, ContrastiveLoss):
            self.training_epoch = partial(
                train_contrastive,
                device=self.device,
                optimizer=self.optimizer,
                criterion=self.loss,
                log_interval=self.log_interval,
            )

        elif isinstance(self.loss, (TripletLoss, nn.TripletMarginLoss)):
            self.training_epoch = partial(
                train_triplet,
                device=self.device,
                optimizer=self.optimizer,
                criterion=self.loss,
                log_interval=self.log_interval,
            )

        else:
            raise Exception(
                "Loss function provided to Trainer constcrutor is not explicitly supported"
            )

    def train(
        self,
        model: nn.Module,
        train_loader,
        scheduler,
        val_loader=None,
        start_epoch: int = 0,
        n_epochs: int = 20,
        dry_run: bool = False,
    ):
        """ """
        for epoch in range(start_epoch):
            scheduler.step

        for epoch in range(start_epoch, n_epochs):
            scheduler.step
            # training stage

            train_loss = self.training_epoch(
                model, train_loader, epoch=epoch, dry_run=dry_run
            )
            if dry_run:
                print("Dry run test completed")
                break
