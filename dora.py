import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

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

        norm = adapted.norm(p=2, dim=0, keepdim=True)
        norm_adapted = adapted / norm
        calc_weights = self.m * norm_adapted
        return F.linear(x, calc_weights, self.bias)

class MoRALayer(nn.Module):
    def __init__(self, d_in, d_out, mora_div=2, weight=None, bias=None):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out

        if weight is not None:
            self.weight = nn.Parameter(weight, requires_grad=False)
        else:
            self.weight = nn.Parameter(torch.Tensor(d_out, d_in), requires_grad=False)

        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = nn.Parameter(torch.Tensor(d_out), requires_grad=False)

        assert d_in % mora_div == 0, f"d_in={d_in} must be divisible by mora_div={mora_div}"
        assert d_out % mora_div == 0, f"d_out={d_out} must be divisible by mora_div={mora_div}"
        self.mora_div = mora_div
        self.m_in = d_in // mora_div
        self.m_out = d_out // mora_div

        self.mora = torch.nn.Parameter(torch.zeros(self.m_in, self.m_out))

        self.dora_mag = nn.Parameter(self.weight.norm(p=2, dim=0, keepdim=True))

        self.compress_type = 1

    def merge(self):
        with torch.no_grad():
            if self.compress_type == 0:
                w = self.mora.repeat(self.mora_div, self.mora_div)
            else:
                w = self.mora.repeat_interleave(self.mora_div, dim=0).repeat_interleave(self.mora_div, dim=1)

            self.weight += w

            self.mora.zero_()

            self.compress_type ^= 1

    def forward(self, x):
        w = self.weight

        if self.compress_type == 0:
            w = w + self.mora.repeat(self.mora_div, self.mora_div)
        else:
            w = w + self.mora.repeat_interleave(self.mora_div, dim=0).repeat_interleave(self.mora_div, dim=1)

        norm_adapted = w / w.norm(p=2, dim=0, keepdim=True)
        w = self.dora_mag * norm_adapted

        return F.linear(x, w, self.bias)


class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim, inner_dim):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, inner_dim)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(inner_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Generating synthetic data
def generate_data(num_samples=100, input_dim=32, output_dim=32):
    X = torch.randn(num_samples, input_dim)
    r = torch.randn(input_dim, output_dim)
    y = torch.matmul(X, r)
    return X, y

# Training function
def train(model, criterion, optimizer, data_loader, epochs=5, enable_merge=False):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if enable_merge:
            merge_weights(model)

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

def replace_linear_with_dora(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Get the input and output dimensions of the current nn.Linear layer
            d_in = module.in_features
            d_out = module.out_features

            # Create a new DoRALayer with the same dimensions
            setattr(model, name, MoRALayer(d_out=d_out, d_in=d_in, weight=module.weight.data.clone(), bias=module.bias.data.clone()))
            #setattr(model, name, DoRALayer(d_out=d_out, d_in=d_in, weight=module.weight.data.clone(), bias=module.bias.data.clone()))
        else:
            # Recursively apply this function to submodules
            replace_linear_with_dora(module)

def merge_weights(model):
    for _, module in model.named_children():
        if isinstance(module, MoRALayer):
            module.merge()
        else:
            # Recursively apply this function to submodules
            merge_weights(module)

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

# Main script
if __name__ == "__main__":
    input_dim, output_dim, inner_dim = 32, 32, 32
    model = SimpleModel(input_dim, output_dim, inner_dim)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    X, y = generate_data(num_samples=1000, input_dim=input_dim, output_dim=output_dim)
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

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
    train(model, criterion, optimizer, data_loader, epochs=20, enable_merge=True)  # Continue training

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(data_loader))
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        print(f"Final (DoRA) Evaluation Loss: {loss.item()}")
