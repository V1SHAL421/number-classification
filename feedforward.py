# MNIST
# DataLoader, Transformation
# Multilayer NN, activation function
# Loss and Optimizer
# Training Loop (batch training)
# Model evaluation
# GPU support

import os
import torch
import torch.nn as nn
import torchvision # for datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # checks if compatible GPU is available

# hyper parameters
input_size = 784 # 28x28 (image size) to be flattened to 1d array
hidden_size = 100 # number of neurons in hidden layer
num_classes = 10 # number of classes in classification task
num_epochs = 3 # number of times the training dataset will be passed through NN during training
batch_size = 100 # number of data samples inputted in nn at once
learning_rate = 0.001 # how much the model's parameters are updated
save_directory = '/models'

# MNIST
try:
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
    transform=transforms.ToTensor()) # dataset transformed into PyTorch tensors
except Exception as e:
    print("Error loading MNIST dataset:", e); exit()

# Split training dataset into training set and validation set
try:
    train_size = int(0.75 * len(train_dataset)) # 75% of data for training set
    valid_size = len(train_dataset) - train_size # 25% of data for validation set
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
except Exception as e:
    print("Error splitting MNIST dataset:", e); exit()
try:
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
    shuffle=True) # loads training dataset into batches

    valid_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
    shuffle=False) # loads validation dataset into batches

    test_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
    shuffle=False) # used for testing and evaluation purposes
except Exception as e:
    print("Error creating data loaders:", e); exit()

try:
    examples = iter(train_loader) # allows you to iterate over the batches of the training dataset
    samples, labels = next(iter(train_loader)) # fetches first batch of data

    print(samples.shape, labels.shape) 

    for i in range(6): # plot
        plt.subplot(2, 3, i+1)
        plt.imshow(samples[i][0], cmap='gray')

    plt.show()
# plots six digits
except Exception as e:
    print("Error displaying example plot:", e) # No need to call exit() since it won't interfere with the execution of the model

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.first_layer = nn.Linear(input_size, hidden_size) # first layer of NN
        self.relu = nn.ReLU() # activation function
        self.second_layer = nn.Linear(hidden_size, num_classes) # second layer of NN

    def forward(self, x): # defines the forward pass of the NN
        out = self.first_layer(x)
        out = self.relu(out)
        out = self.second_layer(out)
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# SummaryWriter object for logging trained statistics

writer = SummaryWriter(log_dir='logging')

# training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
        model.train() # training
        total_train_loss = 0  # Variable to store the total training loss for this epoch

        for i, (images, labels) in enumerate(train_loader): # to get actual index
            # reshape image
            # current image is 100x1x28x28
            # 100x784
            try:
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)

                # forward
                outputs = model(images)
                loss = criterion(outputs, labels)

                # backwards
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()  # Accumulate the training loss for this batch
                writer.add_scalar('Training Loss', loss.item(), epoch * n_total_steps + i)

                if (i+1) % 100 == 0:
                    print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.5f}')
           
            except Exception as e:
                print(f"Error in training loop iteration {i}:", e)

        average_train_loss = total_train_loss / len(train_loader)  # Calculate the average training loss for this epoch
        print(f'Training Loss after epoch {epoch + 1}: {average_train_loss:.5f}')
        try:
            file_path = os.path.join(save_directory, 'trained_model.pth')
            torch.save(model.state_dict(), file_path) # saves the trained model's params to disk
            
        except Exception as e:
            print(f"Error saving the model parameters for epoch {epoch + 1}: {e}")

        model.eval() # validating
        with torch.no_grad():
            total_valid_loss = 0
            for images, labels in valid_loader:
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                outputs = model(images)
                valid_loss = criterion(outputs, labels)
                total_valid_loss += valid_loss.item()

            if (i+1) % 100 == 0:
                print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.5f}')
            average_val_loss = total_valid_loss / len(valid_loader)
            print(f'Validation Loss after epoch {epoch + 1}: {average_val_loss:.5f}')
    # test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    accuracy = 100.0 * n_correct / n_samples
    print(f'accuracy = {accuracy}')

writer.close()

