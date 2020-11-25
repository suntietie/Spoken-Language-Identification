## Sequential Readme 

The model use a GRU model with 4 hidden layers, and a linear layer after it. And there are a few 
points to follow if running the model.

1. In order to get the output of probabilities of 3 classes, a softmax layer should be added after 
the output. 
- Ex.
softmax_out = nn.Softmax(dim=2)

2. Data prepraration after MFCC.

2.1. the intended input data is (batch_size, len_seq, num_features), so a reasonable input is 
(32, 801, 64)

2.2. input label for the model is (batch, len_seq, num_classes), so a reasonable label is 
(32, 801, 3). Label should be one hot, using mapping = dict{’english ’: 0, ’hindi ’: 1, ’mandarin’: 2} 
- Ex. if a frame is labeled as english, then it should be 1 in index 0 and 0 in the other indices.

2.3. Sample code for dataset and dataloader.

- Ex.

class mfcc_set(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        data = self.X[index:index+1]
        # remove dimension 0
        data = data.squeeze(0)
        label = self.y[index]
        return torch.Tensor(data), torch.Tensor(label).type(torch.LongTensor)

eng_loader = DataLoader(eng_test, batch_size=32, shuffle=False)

2.3. Sample code for testing.

- Ex.

### TEST
model.eval()
with torch.no_grad():
    correct = 0.0
    ## using the sequence model
    for x, y in test_loader:
        x = x.to(device)
        y = y[:,-1,:]
        y = torch.argmax(y, axis=1)
        yhat = model(x)[:,-1,:].cpu()

        correct += torch.sum(torch.argmax(yhat, axis=1)==y).double()
        test_acc = correct / len(test_loader.dataset)

