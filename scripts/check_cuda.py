"""Small SLURM preflight for GPU-backed open-model jobs."""

import os
import socket
import sys

import torch


def main():
    print(f"host={socket.gethostname()}")
    print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', 'unset')}")
    print(f"torch.version.cuda={torch.version.cuda}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        current = torch.cuda.current_device()
        print(f"torch.cuda.current_device={current}")
        print(f"torch.cuda.device_name={torch.cuda.get_device_name(current)}")
        return
    raise SystemExit("CUDA is not visible to PyTorch in this SLURM job step.")


if __name__ == "__main__":
    main()
