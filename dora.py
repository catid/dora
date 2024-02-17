import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

torch.manual_seed(0)

# This layer is dropped into your pre-trained PyTorch model where nn.Linear is used
class LoRALayer(nn.Module):
    def __init__(self, d_in, d_out, rank=4, weight=None, bias=None):
        super().__init__()

        if weight is not None:
            self.weight = nn.Parameter(weight, requires_grad=False)
        else:
            self.weight = nn.Parameter(torch.Tensor(d_out, d_in), requires_grad=False)

        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = nn.Parameter(torch.Tensor(d_out), requires_grad=False)

        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.lora_A = nn.Parameter(torch.randn(d_out, rank)*std_dev)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_in))

    def forward(self, x):
        lora = torch.matmul(self.lora_A, self.lora_B)
        return F.linear(x, self.weight + lora, self.bias)

# This layer is dropped into your pre-trained PyTorch model where nn.Linear is used
class DoRALayer(nn.Module):
    def __init__(self, d_in, d_out, rank=4, weight=None, bias=None):
        super().__init__()

        if weight is not None:
            self.weight = nn.Parameter(weight, requires_grad=False)
        else:
            self.weight = nn.Parameter(torch.Tensor(d_out, d_in), requires_grad=False)

        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = nn.Parameter(torch.Tensor(d_out), requires_grad=False)

        # m = Magnitude column-wise across output dimension
        self.m = nn.Parameter(self.weight.norm(p=2, dim=0, keepdim=True))
        
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.lora_A = nn.Parameter(torch.randn(d_out, rank)*std_dev)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_in))

    def forward(self, x):
        lora = torch.matmul(self.lora_A, self.lora_B)
        adapted = self.weight + lora
        column_norm = adapted.norm(p=2, dim=0, keepdim=True)
        norm_adapted = adapted / column_norm
        calc_weights = self.m * norm_adapted
        return F.linear(x, calc_weights, self.bias)

def round_up_sqrt(m):
    sm = int(m ** 0.5 + 0.5)
    while sm*sm > m:
        sm -= 1
    while sm*sm < m:
        sm += 1
    return sm

# Stacked Kronecker-product Layers https://openreview.net/pdf?id=ZjGr1tMVbjw
# Uses 2r*sqrt(nm) parameters instead of nm.
# For for n=512 x m=2048, r must be 256 or less to make it worthwhile.
class SKLinearLayer(nn.Module):
    def __init__(self, d_in, d_out, rank=4, weight=None, bias=None):
        super().__init__()

        if weight is not None:
            self.weight = nn.Parameter(weight, requires_grad=False)
        else:
            self.weight = nn.Parameter(torch.Tensor(d_out, d_in), requires_grad=False)

        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = nn.Parameter(torch.Tensor(d_out), requires_grad=False)

        self.n = d_in
        self.m = d_out
        self.sn = round_up_sqrt(self.n)
        self.np = self.sn * self.sn # n rounded up to next square
        self.sm = round_up_sqrt(self.m)
        self.mp = self.sm * self.sm # m rounded up to next square
        k = self.sn * self.sm

        # Initialize A and B using Kaiming initialization
        self.A = nn.Parameter(torch.empty(k, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)) # a is the parameter for the ReLU
        self.B = nn.Parameter(torch.zeros(rank, k))

    def forward(self, x):
        # Validate that the inputs are of the expected sizes
        if x.size(-1) != self.n:
            raise ValueError("Input vector must have size n")

        S = torch.matmul(self.A, self.B).reshape(self.sn, self.sm, self.sn, self.sm).transpose(2, 1).reshape(self.mp, self.np)
        S_truncated = S[:self.m, :self.n]

        return F.linear(x, self.weight + S_truncated, self.bias)


# Basic MLP feed-forward network like in transformers
class FeedForward(nn.Module):
    def __init__(self, d_in, d_out, mult=4):
        super(FeedForward, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.mult = mult

        hidden_size = d_in * mult
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, d_out)
        )

    def forward(self, x):
        return self.net(x)

# Generating synthetic data
def generate_data(num_samples=100, input_dim=10):
    X = torch.randn(num_samples, input_dim)
    y = torch.sum(X, dim=1, keepdim=True)  # Simple relationship for demonstration
    return X, y

# Training function
def train(model, criterion, optimizer, data_loader, epochs=5):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        #print(f"Epoch {epoch+1}, Loss: {loss.item()}")



def replace_linear_with_dora(model, layer_class):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Get the input and output dimensions of the current nn.Linear layer
            d_in = module.in_features
            d_out = module.out_features

            # Create a new DoRALayer with the same dimensions
            setattr(model, name, layer_class(d_out=d_out, d_in=d_in, weight=module.weight.data.clone(), bias=module.bias.data.clone()))
        else:
            # Recursively apply this function to submodules
            replace_linear_with_dora(module, layer_class)

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

def copy_model(model):
    cloned_model = FeedForward(model.d_in, model.d_out, model.mult)

    cloned_model.load_state_dict(model.state_dict())

    return cloned_model

def continue_training(model, layer_class, name, data_loader):
    model = copy_model(model)

    replace_linear_with_dora(model, layer_class)

    print_model_parameters(model)

    # Continue training with the Dora model
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    print(f"Continuing training with {name} layers...")
    train(model, criterion, optimizer, data_loader, epochs=5)  # Continue training

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(data_loader))
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        print(f"Final ({name}) Evaluation Loss: {loss.item()}")

# Main script
if __name__ == "__main__":
    input_dim, output_dim = 10, 1
    model = FeedForward(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    X, y = generate_data(num_samples=1000, input_dim=input_dim)
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    print_model_parameters(model)

    train(model, criterion, optimizer, data_loader, epochs=100)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(data_loader))
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        print(f"Final Evaluation Loss: {loss.item()}")

    continue_training(model, LoRALayer, "LoRA", data_loader)
    continue_training(model, DoRALayer, "DoRA", data_loader)
    continue_training(model, SKLinearLayer, "SKL", data_loader)
