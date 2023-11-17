import torch
import torch.cuda
import torch.nn as nn
import torchvision
from tqdm.auto import tqdm
from dataset import train_test_val_splitting, SportsDataset
from inception import GoogLeNet
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from saving_loading import saving_model

from train import train_model

SEED = 42

BATCH_SIZE = 128

LR = 1e-3

EPOCHS = 25

transform = transforms.Compose([
    transforms.Resize(size = (224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GoogLeNet(in_channels=3, num_classes=100).to(device)
torch.cuda.manual_seed(SEED)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

image_path = Path("Sports data/")

train_path = image_path / "train"
test_path = image_path / "test"
val_path = image_path / "valid"

train_data, test_data, val_data = train_test_val_splitting(train_path, test_path, val_path, transform)
train_dataload = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, num_workers=0, shuffle = True)
test_dataload = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
val_dataload = DataLoader(dataset=val_data, batch_size = BATCH_SIZE, num_workers=0, shuffle=False)

def main():
    results = train_model(model, train_dataload, test_dataload, optimizer, loss_fn, EPOCHS, device)
    saving_model(model=model, model_name="MyGoogLeNet", saving_path="models")
    plt.plot(results["train_loss"])
    plt.plot(results["train_acc"])
    plt.plot(results["test_loss"])
    plt.plot(results["test_acc"])
    plt.show()

if __name__ == "__main__":
    main()
        

