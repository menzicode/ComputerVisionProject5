# Import statements
import sys
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Returns a training set loader
def train_loader(batch_size):
  tr_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=False)
  return tr_loader

# Returns a testing set loader
def test_loader(batch_size):
  tst_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=False)
  return tst_loader

# Plots images from loader in rows x cols grid 
def plot_images(loader, n_rows, n_cols):
  loader_set = enumerate(loader)
  batch_idx, (img_data, img_targets) = next(loader_set)
  for i in range(n_rows * n_cols):
    plt.subplot(n_rows, n_cols, i+1)
    plt.tight_layout()
    plt.imshow(img_data[i][0], cmap='gray', interpolation='none')
    plt.title("Target Value: {}".format(img_targets[i]))
    plt.xticks([])
    plt.yticks([])
  plt.show()

# Class representing network
# _init_ defines layers 
# forward method applies operations to layers
class MyNetwork(nn.Module):
  def __init__(self):
    super(MyNetwork, self).__init__()
    # Convolution layer with 1 input channel, 10 output channels, and kernal size 5
    self.conv1 = nn.Conv2d(1, 10, 5)
    # Max pooling layer with window size 2
    self.maxp1 = nn.MaxPool2d(2)
    # Convultion layer with 10 input channels, 20 output channels, and kernal size 5
    self.conv2 = nn.Conv2d(10, 20, 5)
    # Dropout layer with default 50%
    self.dropout = nn.Dropout2d()
    # Max pooling layer with window size 2
    self.maxp2 = nn.MaxPool2d(2)
    # Flatten operation
    self.flat = nn.Flatten()
    # Linear layer 1
    self.fc1 = nn.Linear(320, 50)
    # Linear layer 2
    self.fc2 = nn.Linear(50, 10)

  # Feeds input through defined layers and applies operations to layer output
  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(self.maxp1(x))
    x = self.conv2(x)
    x = self.dropout(x)
    x = F.relu(self.maxp2(x))
    x = self.flat(x)
    x = F.relu(self.fc1(x))
    x = F.log_softmax(self.fc2(x))
    return x

# Function to train model
# Parameters are a network object, optimizer object, training set loader, arrays to track losses and counter, and epoch number
def train_model(network, optimizer, trainLoader, train_losses, train_counter, log_interval, epoch): 
  network.train()
  for batch_idx, (data, target) in enumerate(trainLoader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(trainLoader.dataset),
        100. * batch_idx / len(trainLoader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(trainLoader.dataset)))
      # torch.save(network.state_dict(), 'results/model.pth')
      # torch.save(optimizer.state_dict(), 'results/optimizer.pth')

# Function to test model
# Parameters are a trained network, testing set loader, and array to track testing losses
def test_model(network, testLoader, test_losses):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in testLoader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(testLoader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(testLoader.dataset),
    100. * correct / len(testLoader.dataset)))

# Main function 
def main(argv):
  # Handles command line argument

  # Control variables
  torch.backends.cudnn.enabled = False
  # Hyperparameters for model tuning
  
  learning_rate = 0.01
  momentum = 0.5
  log_interval = 10
  seed = 42
  train_batch = 64
  test_batch = 1000
  epochs = 5

  # Parameters for train and test functions
  torch.manual_seed(seed)
  trainLoader = train_loader(train_batch)
  testLoader = test_loader(test_batch)
  network = MyNetwork()
  optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
  train_losses = []
  train_counter = []
  test_losses = []
  test_counter = [i*len(trainLoader.dataset) for i in range(epochs + 1)]  

  # test_model(network, testLoader, test_losses)
  # for epoch in range(1, epochs + 1):
  #   train_model(network, optimizer, trainLoader, train_losses, train_counter, log_interval, epoch)
  #   test_model(network, testLoader, test_losses)

  # fig = plt.figure()
  # plt.plot(train_counter, train_losses, color='blue')
  # plt.scatter(test_counter, test_losses, color='red')
  # plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
  # plt.xlabel('N Training Examples Seen')
  # plt.ylabel('Negative Log Likelihood Loss')
  # plt.show()
  # fig.show()
  plot_images(trainLoader, 2, 3)
  return

if __name__== "__main__":
  main(sys.argv)