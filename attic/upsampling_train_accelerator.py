import time
import os
import random

import logging
import multiprocessing
import argparse

from PIL import Image
from cachetools import LRUCache, cached
import tqdm

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader, random_split

accelerator = Accelerator()

# Configure the logging module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

class ImageUpsamplingDataset(Dataset):
    def __init__(self, img_dir, scale_factor=4, patch_size=64, transform=None, cache_size=int(8000)):  # 8000 images
        self.img_dir = img_dir
        self.scale_factor = scale_factor
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.cache = LRUCache(maxsize=cache_size)
        self.patch_resizer = transforms.Resize((patch_size // scale_factor, patch_size // scale_factor), Image.BICUBIC)
        self.enable_patches = True
        self.patch_size = patch_size
        self.length = len(self.image_filenames)

    def __len__(self):
        return self.length

    def enable_patch_mode(self):
        self.enable_patches = True
        self.cache.clear()

    def enable_full_image_mode(self):
        self.enable_patches = False
        self.cache.clear()

    def load_image(self, idx):
        img_path = os.path.join(self.img_dir, self.image_filenames[idx])
        hr_image = Image.open(img_path).convert("RGB")
        lr_image = hr_image.resize((hr_image.width // self.scale_factor, hr_image.height // self.scale_factor), Image.BICUBIC)
        #logging.info(f"Loaded image: {img_path}")
        return lr_image, hr_image

    def load_random_image_patch(self, idx):
        try:
            img_path = os.path.join(self.img_dir, self.image_filenames[idx])
            hr_image = Image.open(img_path).convert("RGB")

            # Extract random patches from the images
            x = random.randint(0, hr_image.width - self.patch_size)
            y = random.randint(0, hr_image.height - self.patch_size)

            high_res_patch = hr_image.crop((x, y, x + self.patch_size, y + self.patch_size))

            low_res_patch = self.patch_resizer(high_res_patch)

            #logging.info(f"Loaded image patch: {img_path}")
        except Exception as e:
            logging.warning(f"Failed to load/crop {self.image_filenames[idx]}: {e}")
            low_res_patch, high_res_patch = None, None

        return low_res_patch, high_res_patch

    def cached_load_image(self, idx):
        cached = None, None
        try:
            cached = self.cache[idx]
            logging.info(f"Using cached image: {idx}")
        except KeyError:
            if self.enable_patches:
                cached = self.load_random_image_patch(idx)
            else:
                cached = self.load_image(idx)
            self.cache[idx] = cached

        return cached

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError("Sampler is getting out of range")

        #logging.info(f"idx = {idx}, len(self.image_filenames)={len(self.image_filenames)}")
        lr_image, hr_image = self.cached_load_image(idx)

        if self.transform and lr_image is not None and hr_image is not None:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image

# Define the upsampling model
class UpsamplingModel(nn.Module):
    def __init__(self, scale_factor):
        super(UpsamplingModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.interpolation = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.reconstruction = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.interpolation(x)
        x = self.reconstruction(x)
        return x

# Define a custom loss function
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        # Load the pre-trained VGG-19 model
        vgg19 = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT).eval()

        # Choose the layers to compute the perceptual loss
        self.feature_layers = [vgg19.features[i] for i in [1, 6, 11, 20, 29]]

        self.mse_loss = nn.MSELoss()

    def forward(self, generated, target):
        loss = 0
        for layer in self.feature_layers:
            generated = layer(generated)
            target = layer(target)
            loss += self.mse_loss(generated, target)
        return loss

def model_size_bytes(model):
    total_size_bytes = 0
    for param in model.parameters():
        tensor_size_bytes = param.numel() * param.element_size()
        total_size_bytes += tensor_size_bytes
    return total_size_bytes

def tensor_size_in_megabytes(x):
    return x.numel() * x.element_size() // 1000000

class UpsamplingTrainer:
    def __init__(self, img_dir, train_ratio, num_epochs, patience, learning_rate, cached_image_count,
                 scale_factor, patch_size, model_output_path, enable_perceptual_loss, patch_batch_size,
                 full_image_batch_size, plateau_factor):
        self.img_dir = img_dir
        self.train_ratio = train_ratio
        self.num_epochs = num_epochs
        self.patience = patience
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.learning_rate = learning_rate
        self.cached_image_count = cached_image_count
        self.model_output_path = model_output_path
        self.enable_perceptual_loss = enable_perceptual_loss
        self.patch_batch_size = patch_batch_size
        self.full_image_batch_size = full_image_batch_size
        self.plateau_factor = plateau_factor

        self.model = UpsamplingModel(scale_factor=scale_factor)

    def _training_pass(self, batch_size=64, enable_patch_mode=True):
        tensor_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        full_dataset = ImageUpsamplingDataset(self.img_dir, scale_factor=self.scale_factor, patch_size=self.patch_size, transform=tensor_transform)

        if enable_patch_mode:
            full_dataset.enable_patch_mode()
            pin_memory = True
        else:
            full_dataset.disable_patch_mode()
            pin_memory = False

        if self.enable_perceptual_loss:
            loss_fn = PerceptualLoss()
        else:
            loss_fn = nn.MSELoss()

        # In this example, train_ratio is set to 0.8, which means that 80% of the images will
        # be used for training, and the remaining 20% will be used for validation. You can adjust
        # the train_ratio variable to control the size of the training and validation datasets.
        dataset_len = len(full_dataset)
        train_len = int(dataset_len * self.train_ratio)
        val_len = dataset_len - train_len

        train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

        # Now you have train_dataset and val_dataset, which can be used with DataLoader
        loader_workers = multiprocessing.cpu_count() // 4 + 1
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=None, shuffle=True, pin_memory=True, num_workers=loader_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=None, shuffle=False, pin_memory=True, num_workers=loader_workers)

        # Initialize Accelerate
        device = accelerator.device

        # Create the model, loss function, and optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # Prepare for distributed training
        model, optimizer, train_loader, val_loader, loss_fn = accelerator.prepare(self.model, optimizer, train_loader, val_loader, loss_fn)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.patience, factor=self.plateau_factor, verbose=True)

        best_val_loss = float("inf")

        # Training loop
        for epoch in range(self.num_epochs):
            if accelerator.is_main_process:
                logging.info(f"Epoch {epoch + 1}/{self.num_epochs}")

            model.train()

            start_time = time.time()
            running_loss = 0.0

            # Customize the progress bar
            progress_bar = tqdm.tqdm(train_loader)

            with torch.set_grad_enabled(True):
                for batch_idx, (low_res_inputs, high_res_inputs) in enumerate(progress_bar):
                    low_res_inputs = low_res_inputs.to(device)
                    high_res_inputs = high_res_inputs.to(device)

                    if accelerator.is_main_process and batch_idx == 0:
                        logging.info(f"low_res_inputs size: {tensor_size_in_megabytes(low_res_inputs)} MB count={len(low_res_inputs)}")
                        logging.info(f"high_res_inputs size: {tensor_size_in_megabytes(high_res_inputs)} MB count={len(high_res_inputs)}")

                    # Forward pass
                    outputs = model(low_res_inputs)

                    if accelerator.is_main_process and batch_idx == 0:
                        logging.info(f"outputs size: {tensor_size_in_megabytes(outputs)} MB count={len(outputs)}")

                    loss = loss_fn(outputs, high_res_inputs)

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    accelerator.backward(loss)
                    optimizer.step()

                    running_loss += loss.item()

                    progress_bar.set_postfix({"Training Loss": f"{loss.item():.4f}"})

            # Validation
            model.eval()

            val_loss = 0.0

            # Customize the progress bar
            progress_bar = tqdm.tqdm(val_loader)

            with torch.set_grad_enabled(False):
                for batch_idx, (low_res_inputs, high_res_inputs) in enumerate(progress_bar):
                    low_res_inputs = low_res_inputs.to(device)
                    high_res_inputs = high_res_inputs.to(device)

                    outputs = model(low_res_inputs)

                    loss = loss_fn(outputs, high_res_inputs)

                    val_loss += loss.item()

                    progress_bar.set_postfix({"Validation Loss": f"{loss.item():.4f}"})

            end_time = time.time()
            epoch_time = end_time - start_time

            avg_loss = running_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            if accelerator.is_main_process:
                logging.info(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f} seconds")

            scheduler.step(avg_val_loss)

            # Check if validation loss has improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping condition
            if epochs_without_improvement >= self.patience * 2:  # Stop training after patience * 2 epochs without improvement
                logging.info(f"Early stopping at epoch {epoch}, best validation loss: {best_val_loss}")
                break

        logging.info("Training complete.")

    def train(self):
        if accelerator.is_main_process:
            logging.info(f"Model size: {model_size_bytes(self.model)} bytes")

            logging.info("Performing initial training using image patches...")

        self._training_pass(batch_size=self.patch_batch_size, enable_patch_mode=True)

        if accelerator.is_main_process:
            logging.info("Fine-tuning using full images...")

        self._training_pass(batch_size=self.full_image_batch_size, enable_patch_mode=False)

        torch.save(self.model.state_dict(), self.model_output_path)

        if accelerator.is_main_process:
            logging.info("Training complete.  Saved as: {self.model_output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train an upsampling model.")
    parser.add_argument("--img_dir", type=str, default="data", help="Path to the image directory.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Fraction of the dataset used for training.")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs for training.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--scale_factor", type=int, default=2, help="Scale factor for upsampling.")
    parser.add_argument("--patch_size", type=int, default=64, help="Patch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--cached_image_count", type=int, default=8000, help="Number of image pairs to cache in memory.")
    parser.add_argument("--model_output_path", type=str, default="model.pth", help="Path to save the trained model.")
    parser.add_argument("--enable_perceptual_loss", action="store_true", help="Enable the use of perceptual loss during training.")
    parser.add_argument("--patch_batch_size", type=int, default=3500, help="Batch size for patch training steps.")
    parser.add_argument("--full_image_batch_size", type=int, default=1, help="Batch size for full image training steps.")
    parser.add_argument("--plateau_factor", type=float, default=0.1, help="Factor by which the learning rate will be reduced when using ReduceLROnPlateau.")

    args = parser.parse_args()

    trainer = UpsamplingTrainer(
        img_dir=args.img_dir,
        train_ratio=args.train_ratio,
        num_epochs=args.num_epochs,
        patience=args.patience,
        scale_factor=args.scale_factor,
        patch_size=args.patch_size,
        learning_rate=args.learning_rate,
        cached_image_count=args.cached_image_count,
        model_output_path=args.model_output_path,
        enable_perceptual_loss=args.enable_perceptual_loss,
        patch_batch_size=args.patch_batch_size,
        full_image_batch_size=args.full_image_batch_size,
        plateau_factor=args.plateau_factor,
    )

    trainer.train()

if __name__ == "__main__":
    main()
