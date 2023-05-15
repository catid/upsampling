import logging
from tools.logging_tools import setup_colored_logging
setup_colored_logging()

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torchvision.transforms as transforms

from data_loader import DALIDataLoader

class UpsampleNetwork(nn.Module):
    def __init__(self):
        super(UpsampleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.upsample(x)
        x = self.conv3(x)
        return x

def train(args):
    rank = args.rank

    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    torch.cuda.set_device(rank)

    # Load data
    train_loader = DALIDataLoader(batch_size=args.batch_size, num_threads=args.num_threads, device_id=rank)

    # Create model, optimizer and loss function
    model = UpsampleNetwork().cuda(rank)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    criterion = nn.MSELoss().cuda(rank)
    optimizer = optim.Adam(ddp_model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.cuda(rank, non_blocking=True)
            targets = targets.cuda(rank, non_blocking=True)

            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if i % args.print_freq == 0:
                print(f"Rank {rank}, Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}")

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', default=1, type=int, help='Number of GPUs per node')
    parser.add_argument('--rank', default=0, type=int, help='Current execution rank')
    parser.add_argument('--epochs', default=50, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--num_threads', default=4, type=int, help='Number of threads for DALI data loader')
    parser.add_argument('--print_freq', default=100, type=int, help='Print frequency in steps')
    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()


todo:
opt_mod = dynamo.optimize("inductor")(mod)