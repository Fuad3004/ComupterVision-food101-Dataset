
import os 
import torch
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader


NUM_WORKERS=os.cpu_count()
def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS):
  

  # Use ImageFolder to Creare Dataset--->>>

  train_data= datasets.ImageFolder(train_dir, transform=transform)

  test_data= datasets.ImageFolder(test_dir, transform=transform)


  class_names= train_data.classes
  # Turn images into DataLoader

  train_dataloader= DataLoader(
                                dataset=train_data,
                                batch_size= batch_size,
                                shuffle= True,
                                num_workers= NUM_WORKERS,
                                pin_memory=True
  )

  test_dataloader= DataLoader(
                                dataset=train_data,
                                batch_size= batch_size,
                                shuffle= True,
                                num_workers= NUM_WORKERS,
                                pin_memory=True
  )

  return train_dataloader, test_dataloader, class_names
