import torch
import sys
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import os.path
import numpy as np



mode = "standard" # modes = standard, compress, expand


class Network(nn.Module):
    def __init__(self):
        super().__init__() # this superclass call is necesarry!
        self.fc1 = nn.Linear(28*28, 64) #images are 28^2
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 64)
        self.fc4 = nn.Linear(64, 28*28)
    
    def forward(self,x):
        if mode == 'standard':
            x = functional.relu(self.fc1(x))
            x = functional.relu(self.fc2(x))
            x = functional.relu(self.fc3(x))
            x = self.fc4(x)
            return x
        elif mode == 'compress':
            x = functional.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        elif mode == 'expand':
            x = functional.relu(x)
            x = functional.relu(self.fc3(x))
            x = self.fc4(x)
            return x
    


# creates a linear traversal between the point in space represented by two tensors

def traverse(tensor1, tensor2, steps):
    step_length = 1/steps
    result = []
    for i in range (steps+1):
        result.append(torch.add(torch.mul(tensor1,1-(i * step_length)),  torch.mul(tensor2,  (i*step_length))  ))
    return torch.stack(result)







if __name__ == "__main__":
    if sys.argv[1] == 'torch':
        net = Network()

        train = datasets.MNIST('', train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))
        test = datasets.MNIST('', train=False, download=True, transform = transforms.Compose([transforms.ToTensor()]))

        train_set = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
        test_set = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = False)



        if not os.path.isfile('./net_state'):

           

            print(net)
            X = torch.rand((28,28))
            X = X.view(1, 28*28)
            print(net(X))

            optimizer = optim.Adam(net.parameters(), lr=.001)
            
            EPOCHS = 3

            for epoch in range(EPOCHS):
                for data in train_set:
                    #each data batch of training samples
                    X,y = data
                    net.zero_grad()
                    output = net(X.view(-1,28*28))
                    loss = functional.mse_loss(output, X.view(-1,28*28))
                    loss.backward()
                    optimizer.step()
                print(loss)
            
            torch.save(net.state_dict(), "./net_state")
        else:
            net.load_state_dict(torch.load('./net_state'))
        
        """
        correct = 0
        total = 0
        
        with torch.no_grad():
            # possible equivalent to net.train() ... net.eval()
            for data in train_set:
                X,y = data
                output = net(X.view(-1,28*28))
                for idx, i in enumerate(output):
                    if torch.argmax(i) == y[idx]:
                        correct += 1
                    total += 1
        print("ACCURACY: ", round(correct/total, 3))
        """
        if sys.argv[2] == 'comparison':
            mode = 'standard'
            with torch.no_grad():
                for data in train_set:
                    X,y = data
                    print(X[0].shape)
                    output = net(X.view(-1,28*28))
                    for index, value in enumerate(output):
                        plt.imshow(X[index].view(28,28))
                        plt.show()
                        plt.imshow(value.view(28,28))
                        plt.show()
        elif sys.argv[2] == 'traversal':
              with torch.no_grad():
                prev = None
                for data in train_set:
                    X,y = data
                    for index, value in enumerate(X):
                        if prev == None:
                            mode = "compress"
                            prev = net(X[index].view(28*28))
                        else:
                            mode = "compress"
                            curr = net(X[index].view(28*28))
                            print(prev)
                            print(curr)
                            display_tensor = traverse(prev,curr,20)
                            mode = "expand"
                            print('real_sample')
                            for point in display_tensor:
                                image = net(point).view(28,28)
                                plt.imshow(image)
                                plt.show()
                            prev = curr

                        
                        """
                        plt.imshow(X[index].view(28,28))
                        plt.show()
                        plt.imshow(value.view(28,28))
                        plt.show()
                        """

