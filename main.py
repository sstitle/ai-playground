import torch
import torch.nn as nn

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.fc(x)

def main():
    # Check if CUDA is available, otherwise fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Create a simple model and move it to the device
    model = SimpleNet().to(device)
    
    # Print hello world
    print("Hello World from PyTorch!")
    print(f"PyTorch version: {torch.__version__}")

if __name__ == "__main__":
    main()
