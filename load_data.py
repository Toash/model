import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Define data to train the model here
label_data = {
    "apple":"./data/full_numpy_bitmap_apple.npy",
    "banana":"./data/full_numpy_bitmap_banana.npy",
    "bread":"./data/full_numpy_bitmap_bread.npy"
}


class QuickDrawDataset(Dataset):
    def __init__(self,data,labels,transform=None):
        # convert to torch tensor
        self.data = torch.from_numpy(data).to(torch.float32)
        self.labels = torch.from_numpy(labels).to(torch.long)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        data = self.data[idx]
        label = self.labels[idx]
        # Apply existing transformations
        if self.transform:
            data = self.transform(data)
        return data, label
        
def load_data(num_data = 50000, batch_size=16):
    X = []
    y = []
    # Loop through datasets for the labels
    # Each label associated with a number (label_idx)
    for label_idx, file in enumerate(label_data.values()):
        dataset = np.load(file)
        dataset = dataset[:num_data]
        
        # data is np.ndarray shape (784, )
        for data in dataset:
            # Make sure data is ready for loading into model
            data = ((data / 255.0)-0.5)*2
            data = np.reshape(data, (1,28,28))
            X.append(data)
            y.append(label_idx)
            
            
    X = np.array(X)
    y = np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=1,stratify=y)
    
    # Add transformations here
    
    trainset = QuickDrawDataset(X_train, y_train)
    testset = QuickDrawDataset(X_test, y_test)
    
    trainloader = DataLoader(trainset,batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=2)
    
    return (trainset,trainloader,testset,testloader)
        
        
        
        