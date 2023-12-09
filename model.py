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
    
    
def load_model(path = "./model.pth",label_count=3):
    model = Model(label_count)
    model.load_state_dict(torch.load(path))
    return model

def train_model(path="./model.pth",max_iterations=10,batch_size=16,label_count=3):
    torch.manual_seed(0)
    trainset,trainloader,testset,testloader=load_data.load_data()
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    
    for epoch in range(1,max_iterations+1):
        for data in trainloader:
            inputs,labels = data
            optimizer.zero_grad()
            
            outputs = model(inputs)
            print(outputs)
        
        
        
train_model()
