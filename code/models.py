import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset


class MatrixDataset(Dataset):
    def __init__(self, matrices, labels):
        self.matrices = torch.FloatTensor(matrices).unsqueeze(1)  # Add channel dimension
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.matrices[idx], self.labels[idx]

class CNNClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, conv_channels=[1, 16, 32, 64], 
                 fc_units=[576, 128, 2], dropout_rate=0.5):
        super().__init__()
        self.save_hyperparameters()
        
        # Convolutional layers
        conv_layers = []
        for i in range(len(conv_channels)-1):
            conv_layers.extend([
                nn.Conv2d(conv_channels[i], conv_channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(conv_channels[i+1]),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Fully connected layers
        fc_layers = []
        for i in range(len(fc_units)-1):
            fc_layers.extend([
                nn.Linear(fc_units[i], fc_units[i+1]),
                nn.ReLU() if i < len(fc_units)-2 else nn.Identity(),
                nn.Dropout(dropout_rate) if i < len(fc_units)-2 else nn.Identity()
            ])
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # Save intermediate feature maps
        self.feature_maps = {}
        
    def forward(self, x, return_features=False):
        # Store feature maps
        feature_maps = {}
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                feature_maps[f'conv_{i}'] = x
        
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        if return_features:
            return x, feature_maps
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', acc, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

def get_feature_maps(model, sample_input):
    """Helper function to extract feature maps"""
    model.eval()
    with torch.no_grad():
        _, feature_maps = model(sample_input, return_features=True)
    return feature_maps
