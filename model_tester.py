import numpy as np
from load_data import *
from model import *

dataset = np.load(label_data["apple"])
img = dataset[0]
img = torch.from_numpy(img).to(torch.float32)
# data is batched, so need (1,1,28,28) to access the individual data, not (1,28,28))
#   (a,b,c,d) a is batch, b is channel, c and d are dimensions


img = torch.reshape(img,(1,1,28,28))
model = load_model()
prob = get_probabilities(model,img)
print(prob)