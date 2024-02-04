
"""
Contains functionality for creating PyTorch DataLoader's 
for image classification data
"""

import os
from typing import Optional
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,random_split
from pathlib import Path

NUM_WORKERS= os.cpu_count()


def create_dataloaders(
    images_dir:str,
    transform:transforms.Compose,
    batch_size:int,
    test_split_ratio:float,
    num_workers: int=NUM_WORKERS):

    """
    Creates training and testing DataLoaders

    Takes in a training directory and testing directory path and turns them
    into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision trasnforms to perform on training and testing data
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader,test_dataloader,class_names).
        Where class_names is a list of the target classes.
    """

    main_image_folder = datasets.ImageFolder(root=images_dir,
                                             transform=transform
                                            )

    num_train_images= int((1-test_split_ratio)*len(main_image_folder))
    num_test_images= int(test_split_ratio*len(main_image_folder))

    check= (num_test_images+num_train_images) - len(main_image_folder)
    
    if check>=0 or check<0:
        num_test_images=num_test_images+abs(check)
        
    train_data, test_data = random_split(main_image_folder, 
                                        (num_train_images,num_test_images)
                                        )
    
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size= batch_size,
        shuffle=True
        )
    
    test_dataloader= DataLoader(
        dataset=test_data,
        batch_size= batch_size,
        shuffle=False
        )
    
    class_names= main_image_folder.classes

    return train_dataloader,test_dataloader,class_names
