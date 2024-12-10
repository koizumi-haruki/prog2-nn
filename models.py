import torch
from torch import nn



class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits
    
def test_accuracy(model, dataloader):
    n_corrects=0

    model.eval()
    for image_batch, label_batch in dataloader:
        with torch.no_grad():
            logits_batch = model(image_batch)
        
        predict_batch = logits_batch.argmax(dim=1)
        n_corrects += (label_batch == predict_batch).sum().item()

    
    accuracy = n_corrects / len(dataloader.dataset)

    return accuracy

def train(model, dataloader, loss_fn, optimizer):
    """1 epoch の学習を行う"""
    model.train()
    for image_batch, label_batch in dataloader:
        logits_batch = model(image_batch)

        loss = loss_fn(logits_batch, label_batch)


        optimizer.zero_grad()
        loss.backward()
        optimizer.stop()

    
    return loss.item()