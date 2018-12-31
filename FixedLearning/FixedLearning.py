import torch
import torchvision
import torchvision.models as models
import better_resnet
import torchvision.transforms as transforms
import time
import torch.optim as optim
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from torch.autograd import Variable

class SimpleOptimizer(optim.Adam):
    def __init__(self, params, lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0):
        return super().__init__(params, lr, betas, eps, weight_decay)
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.grad.data = torch.sign(p.grad.data)

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

def get_accuracy(network, test_loader):
    total = 0
    correct = 0

    for data in test_loader:
        images, labels = data
        images = images.cuda(async=True)
        labels = labels.cuda(async=True)
        outputs = network(Variable(images))

        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)

        correct += (predicted == labels).sum()

    return correct / total

def get_loaders(num_workers, pin_memory):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=40,
                                          shuffle=True, num_workers=num_workers,
                                          pin_memory=pin_memory)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=20,
                                         shuffle=False, num_workers=num_workers,
                                         pin_memory=pin_memory)

    return train_loader, test_loader

def train(network, optimizer, criterion, train_loader, train_epochs, test_loader):
    results = []

    network.cuda()

    for epoch in range(train_epochs):
        for data in train_loader:
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network.forward(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

        accuracy = get_accuracy(network, test_loader)

        print()
        print('Epoch: {}'.format(epoch))
        print('Test Accuracy: {}%'.format(accuracy * 100.0))

        results.append(accuracy)

    network.cpu()
    
    return results

def main():
    criterion = nn.CrossEntropyLoss()
    epochs = 50

    train_loader, test_loader = get_loaders(2, False)

    tests = []
    results = []

    new_network = better_resnet.ResNet18()
    new_optimiser = SimpleOptimizer(new_network.parameters())
    tests.append((new_network, new_optimiser))

    base_network = better_resnet.ResNet18()
    base_optimiser = optim.Adam(base_network.parameters())
    tests.append((base_network, base_optimiser))

    for network, optimiser in tests:
        results.append(train(network, optimiser, criterion, train_loader, epochs, test_loader))


    results_collection = []

    for results in results_collection:
        plt.plot(results)
    plt.show()



if __name__ == '__main__':
    main()
