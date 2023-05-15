import torch
import argparse

def get_num_parameters(pth_file_path):
    # Load the state_dict from the .pth file
    state_dict = torch.load(pth_file_path, map_location=torch.device('cpu'))

    # Iterate over all entries in the state dict
    for name, tensor in state_dict.items():
        print(f"Name: {name}, Type: {tensor.dtype}, Size: {tensor.size()}")

    # Calculate the total number of parameters
    total_parameters = sum(p.numel() for p in state_dict.values())

    return total_parameters

def main():
    parser = argparse.ArgumentParser(description="Get the number of parameters in a .pth file")
    parser.add_argument("pth_file", help="Path to the .pth file", default="upsampling.pth", nargs="?")

    args = parser.parse_args()

    total_parameters = get_num_parameters(args.pth_file)

    print("Total number of parameters:", total_parameters)

if __name__ == "__main__":
    main()
