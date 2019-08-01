from torchvision import datasets, transforms
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_datasets(dir_path):
    """ Load and return the datasets """
    # Load the data for training, validation, and testing
    train_dir = dir_path + '/train'
    valid_dir = dir_path + '/valid'
    test_dir = dir_path + '/test'
    # Define the transforms for each sets
    train_trans = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_trans = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_trans = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # Load the datasets with ImageFolder and apply the transformations on the images
    train_datasets = datasets.ImageFolder(train_dir, transform=train_trans)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_trans)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_trans)
    return train_datasets, valid_datasets, test_datasets

def get_dataloaders(train_datasets, valid_datasets, test_datasets):
    """ Using the image datasets and the trainforms, define the dataloaders. """
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=False)
    return train_loader, valid_loader, test_loader
