import torch
import torch.nn as nn
import torch.optim as optim

## LSTM Model Definition

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Index hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

## Hyperparameters

input_size = 10  # number of features
hidden_size = 20  # number of features in hidden state
num_layers = 2  # number of stacked lstm layers
output_size = 1  # number of output classes
num_epochs = 100
learning_rate = 0.01

## Create dummy data

# Create random input and target data
X = torch.randn(100, 5, input_size)  # (batch_size, sequence_length, input_size)
y = torch.randn(100, output_size)    # (batch_size, output_size)

## Instantiate the model

model = LSTMModel(input_size, hidden_size, num_layers, output_size)

## Loss and optimizer

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

## Training loop

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

## Test the model

model.eval()
with torch.no_grad():
    test_input = torch.randn(1, 5, input_size)
    test_output = model(test_input)
    print(f'Test Output: {test_output.item():.4f}')