import os
import time
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def main(args):
    # Define the image transformation
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create a dataset using ImageFolder
    dataset = ImageFolder(root=args.input_dir, transform=transform)

    # Create a DataLoader
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    start_time = time.time()
    num_images_processed = 0

    for i, (images, labels) in enumerate(data_loader):
        num_images_processed += len(images)
        
        if time.time() - start_time >= 5:  # Check if at least 5 seconds have passed
            elapsed_time = time.time() - start_time
            images_per_second = num_images_processed / elapsed_time
            print(f"Processed {num_images_processed} images in {elapsed_time:.2f} seconds ({images_per_second:.2f} images/second)")

            # Reset the timer and counter
            start_time = time.time()
            num_images_processed = 0

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a directory of PNG images.")
    parser.add_argument("--input_dir", default="images", help="Input directory containing the images.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the DataLoader.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker threads for DataLoader.")

    args = parser.parse_args()
    main(args)
