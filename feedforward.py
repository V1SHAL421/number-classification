# MNIST
# DataLoader, Transformation
# Multilayer NN, activation function
# Loss and Optimizer
# Training Loop (batch training)
# Model evaluation
# GPU support

import torch
import torch.nn as nn
import torchvision # for datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # checks if compatible GPU is available

# hyper parameters
input_size = 784 # 28x28 (image size) to be flattened to 1d array
hidden_size = 100 # number of neurons in hidden layer
num_classes = 10 # number of classes in classification task
num_epochs = 3 # number of times the training dataset will be passed through NN during training
batch_size = 100 # number of data samples inputted in nn at once
learning_rate = 0.001 # how much the model's parameters are updated

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
    transform=transforms.ToTensor()) # dataset transformed into PyTorch tensors


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
    shuffle=True) # loads training dataset into batches

test_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
    shuffle=False) # used for testing and evaluation purposes

examples = iter(train_loader) # allows you to iterate over the batches of the training dataset
samples, labels = next(iter(train_loader)) # fetches first batch of data

print(samples.shape, labels.shape)

for i in range(6): # plot
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
# plots six digits

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

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): # to get actual index
        # reshape image
        # current image is 100x1x28x28
        # 100x784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.5f}')
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

