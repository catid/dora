import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.lora_A = nn.Parameter(torch.zeros(d_out, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, d_in))

    def forward(self, x):
        lora = torch.matmul(self.lora_A, self.lora_B)
        adapted = self.weight + lora
        column_norm = adapted.norm(p=2, dim=0, keepdim=True)
        norm_adapted = adapted / column_norm
        calc_weights = self.m * norm_adapted
        return F.linear(x, calc_weights, self.bias)


class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        return x

class SimpleModelDora(nn.Module):
    def __init__(self, output_dim, input_dim):
        super(SimpleModelDora, self).__init__()
        self.layer1 = DoRALayer(output_dim, input_dim)

    def forward(self, x):
        x = self.layer1(x)
        return x

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



def replace_linear_with_dora(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Get the input and output dimensions of the current nn.Linear layer
            d_in = module.in_features
            d_out = module.out_features

            # Create a new DoRALayer with the same dimensions
            setattr(model, name, DoRALayer(d_out=d_out, d_in=d_in, weight=module.weight.data.clone(), bias=module.bias.data.clone()))
        else:
            # Recursively apply this function to submodules
            replace_linear_with_dora(module)

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

# Main script
if __name__ == "__main__":
    input_dim, output_dim = 10, 1
    model = SimpleModel(input_dim, output_dim)
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

    replace_linear_with_dora(model)

    print_model_parameters(model)

    # Continue training with the Dora model
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    print("Continuing training with DoRA layers...")
    train(model, criterion, optimizer, data_loader, epochs=5)  # Continue training

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(data_loader))
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        print(f"Final (DoRA) Evaluation Loss: {loss.item()}")
