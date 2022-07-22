#!/usr/bin/python3

import torch
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms, utils
from datetime import datetime
from typing import Optional
import argparse
import os


class LightningMNISTClassifier(pl.LightningModule):

    def __init__(self, learning_rate=2e-4):
        super(LightningMNISTClassifier, self).__init__()

        self.learning_rate = learning_rate

        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()        
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.relu(x)
        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)
        return x   

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        return avg_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 1, num_workers: int = 1, pin_memory: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)
        

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', default=1, type=int, help='number of nodes used')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--batchsize', default=1, type=int, help='number of samples per batch')
    parser.add_argument('--numworkers', default=1, type=int, help='number of workers per node')
    parser.add_argument('--epochs', default=2, type=int, help='number of epochs to run')
    args = parser.parse_args()

    print('pytorch: ' + torch.__version__)
    print('pytorch lightning: ' + pl.__version__)
    print('args.nodes: ' + str(args.nodes))

    data_module = MNISTDataModule(data_dir='./', batch_size=args.batchsize, num_workers=args.numworkers)
    model = LightningMNISTClassifier()
    trainer = pl.Trainer(num_nodes=args.nodes, devices=args.gpus, accelerator="gpu", strategy="ddp", max_epochs=args.epochs, log_every_n_steps=1)

    start = datetime.now()
    trainer.fit(model, data_module)
    print("Training complete in: " + str(datetime.now() - start))



if __name__ == '__main__':
    main()



# export NCCL_SOCKET_IFNAME=eno1 MASTER_PORT=1234 MASTER_ADDR=192.168.10.10 NODE_RANK=0 && python3 dist_train.py --nodes 2 --gpus 1 --batchsize 100 --epochs 1 --numworkers 2

# export NCCL_SOCKET_IFNAME=enp4s0 MASTER_PORT=1234 MASTER_ADDR=192.168.10.10 NODE_RANK=1 && python3 dist_train.py --nodes 2 --gpus 1 --batchsize 100 --epochs 1 --numworkers 2