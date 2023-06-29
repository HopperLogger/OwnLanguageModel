import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from PyQt6.QtCore import QThread, pyqtSignal

class TrainModel(QThread):
    increaseProgressBar = pyqtSignal(float)
    sendLogMessage = pyqtSignal(str, str)
    updateStats = pyqtSignal(str)

    def __init__(self, epochs, batch_size, ltsm, training_test_size, learning_rate, seq_length, samples_to_keep, gpu_enable):
        QThread.__init__(self)
        self.epochs = epochs
        self.batch_size = batch_size
        self.ltsm = ltsm
        self.training_test_size = training_test_size
        self.learning_rate = learning_rate
        self.seq_length = seq_length
        self.samples_to_keep = samples_to_keep
        self.gpu_enable = gpu_enable

    def loadDataChunks(self, file_name: str) -> list:
        """
        Loads data in chunks from the ProcessedData folder.

        Parameters:\n
            file_name (str): The name of the files to load.

        Returns:\n
            list: The loaded data.
        """
        num_files = len([name for name in os.listdir(f'{os.getcwd()}/ProcessedData') if name.startswith(file_name)])
        arr = []
        for i in range(0, num_files):
            chunk = np.load(f'{os.getcwd()}/ProcessedData/{file_name}-{i}.npz')
            arr += list(chunk.values())
        
        return arr

    def run(self):
        self.sendLogMessage.emit('Loading training data...', 'yellow')
        training_data = self.loadDataChunks('pos-encoded-training-data')
        self.sendLogMessage.emit('Loaded training data.', 'green')
        self.sendLogMessage.emit('Creating training data tensor...', 'yellow')
        #training_data = [[np.array([0.5,1,0.43,0.1,0.2]),np.array([0.2,0.1,0.43,0.4,0.33]),np.array([0.6,1,0.46,0.7,0.9])],[np.array([0.3,1,0.41,0.09,0.33]),np.array([0.45,1,0.4,0.1,0.76]),np.array([0.21,1,0.33,0.45,0.66])]]
        training_data = pad_sequence([torch.tensor(sentence) for sentence in training_data], batch_first=True, padding_value=0)
        #training_data_tensor = torch.tensor(training_data, dtype=torch.float32, device='cuda' if self.gpu_enable else 'cpu') # Convert the training data to a tensor
        training_data_tensor = training_data.clone().detach().to(dtype=torch.float32, device='cuda' if self.gpu_enable else 'cpu')
        training_data_dataset = TensorDataset(training_data_tensor) # Create a TensorDataset from the training data tensor
        training_data_dataloader = DataLoader(training_data_dataset, batch_size=self.batch_size, shuffle=True) # Create a DataLoader to handle batching and shuffling
        self.sendLogMessage.emit('Created training data tensor.', 'green')
        
        class TransformerModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
                super(TransformerModel, self).__init__()

                # Embedding layer
                self.embedding = nn.Linear(input_dim, hidden_dim)

                # Transformer layers
                self.transformer_layers = nn.ModuleList([
                    TransformerLayer(hidden_dim, num_heads)
                    for _ in range(num_layers)
                ])

                # Output layer
                self.output_layer = nn.Linear(hidden_dim, input_dim)

            def forward(self, x):
                x = self.embedding(x)

                for layer in self.transformer_layers:
                    x = layer(x)

                output = self.output_layer(x)
                return output


        class TransformerLayer(nn.Module):
            def __init__(self, hidden_dim, num_heads):
                super(TransformerLayer, self).__init__()

                self.self_attention = MultiHeadAttention(hidden_dim, num_heads)
                self.feed_forward = FeedForward(hidden_dim)

                self.layer_norm1 = nn.LayerNorm(hidden_dim)
                self.layer_norm2 = nn.LayerNorm(hidden_dim)

            def forward(self, x):
                residual = x

                x = self.layer_norm1(x + self.self_attention(x))
                x = self.layer_norm2(x + self.feed_forward(x))

                output = x + residual
                return output


        class MultiHeadAttention(nn.Module):
            def __init__(self, hidden_dim, num_heads):
                super(MultiHeadAttention, self).__init__()

                self.hidden_dim = hidden_dim
                self.num_heads = num_heads
                self.head_dim = hidden_dim // num_heads

                self.q_linear = nn.Linear(hidden_dim, hidden_dim)
                self.k_linear = nn.Linear(hidden_dim, hidden_dim)
                self.v_linear = nn.Linear(hidden_dim, hidden_dim)
                self.output_linear = nn.Linear(hidden_dim, hidden_dim)

            def forward(self, x):
                batch_size, seq_len, hidden_dim = x.size()

                # Split the hidden dimension into heads
                q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
                k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
                v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

                # Transpose dimensions for matrix multiplication
                q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
                k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
                v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

                # Compute scaled dot-product attention
                attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))
                attention_weights = torch.softmax(attention_scores, dim=-1)
                attention_output = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len, head_dim)

                # Transpose dimensions back to (batch_size, seq_len, hidden_dim)
                attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)

                # Apply output linear layer
                output = self.output_linear(attention_output)
                return output

        class FeedForward(nn.Module):
            def __init__(self, hidden_dim, ff_dim=2048):
                super(FeedForward, self).__init__()

                self.linear1 = nn.Linear(hidden_dim, ff_dim)
                self.linear2 = nn.Linear(ff_dim, hidden_dim)

            def forward(self, x):
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.linear2(x)
                return x

        # Instantiate your language model
        model = TransformerModel(input_dim=100, hidden_dim=10, num_heads=1, num_layers=1)

        loss_fn = nn.MSELoss() # Defining the loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate) # Defining the optimizer

        # Create the targets
        target_data_tensor = training_data_tensor.roll(-1, dims=1)
        target_data_dataset = TensorDataset(target_data_tensor) # Create a TensorDataset from the training data tensor
        target_data_dataloader = DataLoader(target_data_dataset, batch_size=self.batch_size, shuffle=True) # Create a DataLoader to handle batching and shuffling

        # Training loop
        self.sendLogMessage.emit("Training model...", "yellow")
        for epoch in range(self.epochs):
            for batch, target_batch in zip(training_data_dataloader, target_data_dataloader):
                # Clear accumulated gradients
                optimizer.zero_grad()

                # Get inputs and targets from the batch
                inputs = batch[0]
                targets = target_batch[0]

                # Forward pass
                outputs = model(inputs)

                # Compute the loss
                loss = loss_fn(outputs, targets)

                # Backpropagation
                loss.backward()

                # Update the parameters
                optimizer.step()

            # Print the loss after each epoch
            self.sendLogMessage.emit(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}", "blue")
