#import torch
#print("CUDA available:", torch.cuda.is_available())
#print("PyTorch version:", torch.__version__)
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
