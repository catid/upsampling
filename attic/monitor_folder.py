import os
import time
import argparse

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            file_path = os.path.join(dirpath, f)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    return total_size

def monitor_folder_rate(folder_path, interval):
    while True:
        initial_size = get_folder_size(folder_path)
        time.sleep(interval)
        final_size = get_folder_size(folder_path)
        size_difference = final_size - initial_size
        rate = size_difference / interval
        print(f"Data rate: {rate/1_000_000.0} Mbytes/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate data rate of a folder.")
    parser.add_argument("folder", help="Path to the folder to monitor")
    parser.add_argument("-i", "--interval", type=int, default=4, help="Monitoring interval in seconds (default: 4)")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print("The specified folder does not exist.")
        exit(1)

    monitor_folder_rate(args.folder, args.interval)
