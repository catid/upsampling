import time
import torch
import argparse

def benchmark_device(device, matrix_size=1024, n_iterations=50, multiplications_per_iteration=10):
    torch.cuda.empty_cache()

    # Create random matrices on the specified device
    matrix_a = torch.randn(matrix_size, matrix_size, device=device)
    matrix_b = torch.randn(matrix_size, matrix_size, device=device)

    # Warm-up run
    for _ in range(multiplications_per_iteration):
        torch.matmul(matrix_a, matrix_b)

    # Benchmark
    start_time = time.time()
    for _ in range(n_iterations):
        for _ in range(multiplications_per_iteration):
            torch.matmul(matrix_a, matrix_b)
    torch.cuda.synchronize(device)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time for {n_iterations * multiplications_per_iteration} matrix multiplications on {device}: {elapsed_time:.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a CUDA device for machine learning tasks")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device identifier (default: cuda:0)")
    parser.add_argument("--matrix_size", type=int, default=1024, help="Size of the square matrices for benchmarking (default: 1024)")
    parser.add_argument("--n_iterations", type=int, default=50, help="Number of iterations for the benchmark (default: 50)")
    parser.add_argument("--multiplications_per_iteration", type=int, default=10, help="Number of matrix multiplications per iteration (default: 10)")
    args = parser.parse_args()

    device = torch.device(args.device)
    benchmark_device(device, args.matrix_size, args.n_iterations, args.multiplications_per_iteration)
