import time
from PIL import Image
import numpy as np
import io
import webp


def benchmark_png(image_path, num_iterations=10):
    # Load the image
    img = Image.open(image_path)

    # Benchmark the PNG encoder
    start_time = time.time()
    for i in range(num_iterations):
        img.save("test.png", format='PNG')
    elapsed_time = time.time() - start_time

    # Print the results
    print(f'PNG: {num_iterations} iterations in {elapsed_time:.3f} seconds')


def benchmark_webp(image_path, num_iterations=10):
    # Load the image
    img = Image.open(image_path)

    # Benchmark the WebP encoder
    start_time = time.time()
    for i in range(num_iterations):
        webp.save_image(img, "test.webp", lossless=True)
    elapsed_time = time.time() - start_time

    # Print the results
    print(f'WebP: {num_iterations} iterations in {elapsed_time:.3f} seconds')


if __name__ == '__main__':
    image_path = 'hd_image.png'
    benchmark_webp(image_path)
    benchmark_png(image_path)

