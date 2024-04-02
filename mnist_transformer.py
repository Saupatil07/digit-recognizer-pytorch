import torch
import torchvision
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split
from torchvision import transforms
from vit import ViT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('train.csv')
X, y = data.iloc[:, 1:], data['label']
X = X.to_numpy()
y = y.to_numpy()

X = X / 255

# Convert numpy arrays to torch tensors:

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

print(f'Previous X dimensions: {X_tensor.shape}')

# Get input X in [N, C, H, W] format for input to neural network
X_tensor = X_tensor.reshape(X_tensor.shape[0], 28, 28)
X_tensor = torch.unsqueeze(X_tensor, 1)

print(f'New X dimensions: {X_tensor.shape}')

# Create PyTorch Dataset using TensorDataset(input_features, labels):
mnist_dataset = TensorDataset(X_tensor, y_tensor)

training_set, validation_set = random_split(mnist_dataset, [36000, 6000], generator=torch.Generator().manual_seed(1))

batch_size = 64
torch.manual_seed(1)

train_dl = DataLoader(training_set, batch_size, shuffle=True)
valid_dl = DataLoader(validation_set, batch_size, shuffle=True)

v = ViT(
    image_size = 28,
    patch_size = 4,
    num_classes = 10,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    channels=1
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(v.parameters(), lr=0.001)
num_epochs = 20

def train(model, num_epochs, train_dl, valid_dl):
    
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    
    for epoch in range(num_epochs):
        
        model.train()
        
        for x_batch, y_batch in train_dl:
            
            pred = model(x_batch)    # Get predicted values for this batch     
            loss = loss_fn(pred, y_batch)    # Calculate the loss
            loss.backward()    # Compute gradients
            optimizer.step()    # Update parameters
            optimizer.zero_grad()    # Reset grads to zero
            
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)    # Total loss for the whole batch
            
            # Each image in batch has predicted class probabilities for classes 0 to 9.
            # Take highest class probability (our predicted label) and check if it equals true label.
            # Therefore is_correct = 1 if image prediction is correct, 0 otherwise.
            
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            
            accuracy_hist_train[epoch] += is_correct.sum()    # Number of correctly predicted images in this batch
        
        
        loss_hist_train[epoch] /= len(train_dl.dataset)    # After accumulating total losses from each batch, calculate loss per sample.
        accuracy_hist_train[epoch] /= len(train_dl.dataset)    # After getting total number of correct preds across all batches, calculate accuracy per sample (so between 0 and 1). 
    
    
        
        # Put model in eval mode for evaluating on validation set
        model.eval()
        
        with torch.no_grad():     # Disable gradient calculation, since we do not need this for getting validation set accuracy and loss
            
            for x_batch, y_batch in valid_dl:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum()
                
        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)
        
        print(f'Epoch {epoch+1} accuracy: {accuracy_hist_train[epoch]:.4f}    val_accuracy: {accuracy_hist_valid[epoch]:.4f} ')
        
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid


initial_hist = train(v, num_epochs, train_dl, valid_dl)