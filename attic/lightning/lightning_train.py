import argparse
import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
from torch.utils.data.dataset import random_split

class PyTorchNiN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.classifier(x)
        logits = x.view(x.size(0), self.num_classes)
        return logits


# LightningModule that receives a PyTorch model as input
class LightningModel(pl.LightningModule):
    def __init__(self, model, num_classes, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        # The inherited PyTorch module
        self.num_classes = num_classes
        self.model = model

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=['model'])

        # Set up attributes for computing the accuracy
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)

    # Defining the forward method is only necessary 
    # if you want to use a Trainer's .predict() method (optional)
    def forward(self, x):
        return self.model(x)
        
    # A common forward step to compute the loss and labels
    # this is used for training, validation, and testing below
    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)
        loss = torch.nn.functional.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)

        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss, sync_dist=True)
        
        # To account for Dropout behavior during evaluation
        self.model.eval()
        with torch.no_grad():
            _, true_labels, predicted_labels = self._shared_step(batch)
        self.train_acc.update(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False, sync_dist=True)
        self.model.train()
        return loss  # this is passed to the optimzer for training

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("valid_loss", loss, sync_dist=True)
        self.valid_acc(predicted_labels, true_labels)
        self.log("valid_acc", self.valid_acc,
                 on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class DataModule(pl.LightningDataModule):
    def __init__(self, data_path='./', batch_size=128, num_workers=12):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_path,
                         download=True)

    def setup(self, stage=None):
        self.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),                
            torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train = datasets.CIFAR10(root=self.data_path, 
                                 train=True, 
                                 transform=self.train_transform,
                                 download=False)

        self.test = datasets.CIFAR10(root=self.data_path, 
                                     train=False, 
                                     transform=self.test_transform,
                                     download=False)

        self.train, self.valid = random_split(train, lengths=[45000, 5000])

    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train, 
                                  batch_size=self.batch_size, 
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(dataset=self.valid, 
                                  batch_size=self.batch_size, 
                                  drop_last=False,
                                  shuffle=False,
                                  num_workers=self.num_workers)
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(dataset=self.test, 
                                 batch_size=self.batch_size, 
                                 drop_last=False,
                                 shuffle=False,
                                 num_workers=self.num_workers)
        return test_loader


def main(args):

    # Settings
    torch.set_float32_matmul_precision("high")

    # Logger
    logger = TensorBoardLogger(save_dir="logs/", name="cifar_experiment")

    torch.manual_seed(1) 
    data_module = DataModule(
        data_path='./lightning_data',
        num_workers=args.num_workers,
        batch_size=args.batch
    )

    num_classes = 10

    pytorch_model = PyTorchNiN(num_classes=num_classes)

    lightning_model = LightningModel(
        model=pytorch_model,
        num_classes=num_classes,
        learning_rate=1e-3
    )

    callbacks = [ModelCheckpoint(
        save_top_k=1, mode='max', monitor="valid_acc")]  # save top 1 model 

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        accelerator=args.accelerator,
        num_nodes=args.num_nodes,
        devices=args.devices,
        logger=logger,
        strategy=args.strategy,
        log_every_n_steps=100)

    start_time = time.time()

    trainer.fit(model=lightning_model, datamodule=data_module)

    runtime = (time.time() - start_time)/60

    if trainer.global_rank == 0:
        print(f"Training took {runtime:.2f} min in total.")


if __name__ == '__main__':
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='PyTorch Lightning training script')
    parser.add_argument('--devices', default="auto", help='Number of GPUs per node')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=256, help='Number of training epochs')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of loader workers')
    parser.add_argument('--accelerator', type=str, default="auto", choices=['auto', 'cpu', 'cuda', 'hpu', 'ipu', 'mps', 'tpu'], help='Distributed backend')
    parser.add_argument('--strategy', type=str, default="auto", choices=['ddp', 'ddp2', 'deepspeed'], help='Distributed backend')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes (machines) for multi-node training')
    parser.add_argument('--node-rank', type=int, default=0, help='Rank of the current node (machine) for multi-node training')
    parser.add_argument('--master-addr', type=str, default='127.0.0.1', help='IP address of the master node for multi-node training')
    parser.add_argument('--master-port', type=str, default='8888', help='Port number of the master node for multi-node training')
    args = parser.parse_args()

    main(args)
