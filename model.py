import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import load_data

class Model(nn.Module):
    def __init__(self, label_count = 3) -> None:
        super().__init__()
        # Define model layers (implicitly contains the parameters)
        # Convolutional layers extract features. 
        # These list of features are then used to predict the class. 
        
        # https://datascience.stackexchange.com/questions/37975/what-is-the-intuition-behind-using-2-consecutive-convolutional-filters-in-a-conv
        
        self.conv1 = nn.Conv2d(1,6,5) # first conv layer extracts low-level features
        self.pool = nn.MaxPool2d(2,2) # downsampling, increase translation invariance
        self.conv2 = nn.Conv2d(6,16,3) # second conv layer extracts higher-level features
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, label_count) # output layer
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1) # flatten each datapoint
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
"""
Returns probabilies for classes for a given input
"""
def get_probabilities(model, input):
    model.eval()
    output = model(input)
    return tuple(output)
    
def load_model(path = "./model.pth",label_count=3):
    model = Model(label_count)
    model.load_state_dict(torch.load(path))
    return model

"""
Trains model and then saves it
"""
def train_model(path="./model.pth",max_iterations=10, num_data=10000):
    torch.manual_seed(0)
    trainset,trainloader,testset,testloader=load_data.load_data(num_data=num_data)
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    
    for epoch in range(1,max_iterations+1):
        epoch_loss = 0
        epoch_correct_count = 0
        #mini batch
        for data in trainloader:
            inputs,labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs,labels)
            # Compute gradient of loss with respect to weights and biases
            loss.backward() 
            optimizer.step()
        
            epoch_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(outputs,1)
            epoch_correct_count += (predicted == labels).sum().item()
            
        print(f'Epoch {epoch} loss: {epoch_loss / len(trainloader.dataset)}')
        print(f'Epoch {epoch} training accuracy: {epoch_correct_count/len(trainloader.dataset)}')
    
    # save the model
    torch.save(model.state_dict(),path)
    test_model(model,testloader,criterion)
        
def test_model(model:nn.Module, testloader,criterion:nn.CrossEntropyLoss):
    model.eval()
    
    test_loss=0
    test_correct=0
    for data in testloader:
        inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        test_loss+=loss.item()
        
        _, predicted = torch.max(outputs,1)
        test_correct += (predicted == labels).sum().item()
    
    print("Testing loss:", test_loss/len(testloader.dataset))
    print("Testing accuracy: ", test_correct/len(testloader.dataset))

if __name__ == '__main__': 
    train_model(max_iterations=15,num_data=10000)
