from typing import Callable, List, Optional, Tuple, Union

import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from models import model
from visualizations import visualize

def validation(
    model: nn.Module,
    testloader: torch.utils.data.DataLoader,
    criterion: Union[Callable, nn.Module],
) -> Tuple[float, float]:
    """Validation pass through the dataset."""
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:
        images = images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = labels.data == ps.max(1)[1]
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean().item()

    return test_loss, accuracy

def train(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    criterion: Union[Callable, nn.Module],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epochs: int = 5,
    print_every: int = 40,
) -> None:
    """Train a PyTorch Model."""
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    steps = 0
    running_loss = 0
    tr_loss = []
    te_loss = []
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1

            # Flatten images into a 784 long vector
            images.resize_(images.size()[0], 784)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)

                print(
                    "Epoch: {}/{}.. ".format(e + 1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                    "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
                    "Test Accuracy: {:.3f}".format(accuracy / len(testloader)),
                )

                tr_loss.append(running_loss / print_every)
                te_loss.append(test_loss / len(testloader))

                running_loss = 0

                # Make sure dropout and grads are on for training
                model.train()
    
    visualize.make_visualizations(tr_loss,te_loss)
    

if __name__=="__main__":
    model = model.Network(784, 10, [512, 256, 128])
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    # Download and load the training data
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    train(model = model,
          trainloader = trainloader,
          testloader = testloader,
          criterion = criterion,
          optimizer = optimizer,
          epochs = 5)

    print("Saving model...")
    
    if not os.path.exists("./models/runs/"):
        os.mkdir("./models/runs/")
    
    torch.save(model.state_dict(),'./models/runs/my_model.pth')